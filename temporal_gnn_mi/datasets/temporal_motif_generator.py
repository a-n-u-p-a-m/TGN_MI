import torch
import random
from typing import List, Tuple, Dict, Any
from torch_geometric.data import TemporalData, Batch
from torch.utils.data import Dataset

from temporal_gnn_mi.config import DatasetConfig


def sort_events_by_time(events: List[Tuple]) -> List[Tuple]:
    """Sorts a list of events by their timestamp (index 2)."""
    return sorted(events, key=lambda x: x[2])


def _has_candidate_motif(events: List[Tuple], A: int, B: int, C: int,
                         time_window_delta: float) -> bool:
    """Check whether the candidate-specific motif A→B→C completes within Δ.

    Looks for a pair of edges (A→B @ t1, B→C @ t2) where 0 < t2 - t1 < Δ.
    This is candidate-specific: it only checks the designated (A, B, C) nodes,
    NOT arbitrary u→v→w chains elsewhere in the graph.

    The task definition is:
        y = 1  ⟺  A→B→C completes within Δ
    so only the candidate triple matters.
    """
    # Collect all A→B and B→C edges with their timestamps
    ab_times = [e[2] for e in events if e[0] == A and e[1] == B]
    bc_times = [e[2] for e in events if e[0] == B and e[1] == C]

    for t1 in ab_times:
        for t2 in bc_times:
            gap = t2 - t1
            if 0 < gap < time_window_delta:
                return True
    return False


def _count_candidate_motifs(events: List[Tuple], A: int, B: int, C: int,
                            time_window_delta: float) -> int:
    """Count how many distinct A→B→C completions exist within Δ.

    Used for positive sample validation: we want exactly 1 valid completion
    (the planted one). Multiple completions can arise from the type-2 distractor
    or accidental edges.
    """
    ab_times = [e[2] for e in events if e[0] == A and e[1] == B]
    bc_times = [e[2] for e in events if e[0] == B and e[1] == C]

    count = 0
    for t1 in ab_times:
        for t2 in bc_times:
            gap = t2 - t1
            if 0 < gap < time_window_delta:
                count += 1
    return count


class TemporalMotifDataset(Dataset):
    """
    Generates synthetic temporal graphs with planted temporal chain motifs
    for a binary classification task.
    
    ── Research Motivation ──
    To study how Temporal Graph Neural Networks (TGNNs) learn temporal patterns,
    we construct a synthetic task where a specific temporal chain motif A→B→C 
    must complete within a time window Δ (i.e., t₂ − t₁ < Δ).
    
    Label 1 = motif completes within Δ
    Label 0 = motif is broken (topology, timing, or deletion)
    
    ── Task Definition ──
    The task is candidate-specific motif completion:
        y = 1  ⟺  the designated triple (A, B, C) has A→B→C within Δ
    The model receives candidate_nodes = [A, B, C] and classifies from
    memory(A) + memory(B) + memory(C). Arbitrary u→v→w chains elsewhere
    in the graph are irrelevant to the label.

    ── Anti-Shortcut Design ──
    The dataset prevents trivial shortcuts via:
    1. Background noise:  5-10× more noise edges than motif edges
    2. Distractors:       correct timing / wrong topology, and vice versa
    3. Partial motifs:    chain-starts that never complete
    4. Node-identity:     shared node B must match between the two edges
    5. Event-count parity: all samples are padded with noise to have exactly the same number of global events to prevent graph-size data leaks.
    
    ── Corruption Strategies for Negative Samples ──
    Each negative sample is created by taking a valid positive sample and 
    applying ONE minimal corruption:
      A) Topology break  — change B→C to D→C (shared node mismatch)
      B) Timing break    — push t₂ outside the time window Δ
      C) Edge deletion   — replace the completing edge with random noise
      D) Source shuffle   — change A→B to X→B (origin mismatch)
      E) Temporal misordering — swap t₁ and t₂
    """
    
    def __init__(self, config: DatasetConfig, num_samples: int = 1000, split: str = 'train'):
        self.config = config
        self.num_samples = num_samples
        self.split = split
        self.max_time = 100.0
        self.feature_dim = 16
        
        # Corruption strategies for negative samples
        self.corruption_strategies = [
            'topology_break',        # Strategy A
            'timing_break',          # Strategy B
            'edge_deletion',         # Strategy C
            'source_shuffle',        # Strategy D
            'temporal_misordering',  # Strategy E
        ]
        
        # Generate balanced dataset
        self.samples = []
        num_pos = num_samples // 2
        num_neg = num_samples - num_pos
        
        for _ in range(num_pos):
            self.samples.append(self._generate_positive_sample())
        for _ in range(num_neg):
            self.samples.append(self._generate_negative_sample())
            
        random.shuffle(self.samples)

    # ──────────────────────────────────────────────────
    #  Helper methods
    # ──────────────────────────────────────────────────
    
    def _random_feature(self) -> torch.Tensor:
        """Random edge feature vector of dimension self.feature_dim."""
        return torch.randn(self.feature_dim)

    def _pick_other_node(self, exclude: set) -> int:
        """Pick a random node NOT in the exclude set."""
        candidates = [n for n in range(self.config.num_nodes) if n not in exclude]
        if not candidates:
            return random.randint(0, self.config.num_nodes - 1)
        return random.choice(candidates)

    # ──────────────────────────────────────────────────
    #  Anti-shortcut noise injection
    # ──────────────────────────────────────────────────
    
    def _add_noise_events(self, events: List, num_noise: int):
        """Adds background noise edges between random nodes at random times."""
        for _ in range(num_noise):
            u = random.randint(0, self.config.num_nodes - 1)
            v = random.randint(0, self.config.num_nodes - 1)
            while u == v:
                v = random.randint(0, self.config.num_nodes - 1)
            t = random.uniform(0, self.max_time)
            events.append((u, v, t, self._random_feature()))

    def _add_distractors(self, events: List, nodes_dict: Dict):
        """
        Adds distractor edges that look like motifs but aren't:
        1. Correct timing, wrong topology  (random chain within Δ)
        2. Correct topology, wrong timing  (real A→B→C but t₂ − t₁ > Δ)

        Guard: Type-1 distractors must not create edges that could form part
        of a candidate A→B→C chain (i.e. no A→B or B→C edges). Type-2
        distractors use timing that cannot pair with the planted motif to
        create a second within-Δ completion.
        """
        A = nodes_dict.get('A', -1)
        B = nodes_dict.get('B', -1)
        C = nodes_dict.get('C', -1)

        for _ in range(self.config.num_distractors):
            # ─── Type 1: wrong topology, correct timing ───
            # Avoid creating ANY edge that could be part of A→B→C:
            # neither (u=A, v=B) nor (v=B, w=C)
            for _guard in range(20):
                u, v, w = random.sample(range(self.config.num_nodes), 3)
                if not ((u == A and v == B) or (v == B and w == C)):
                    break
            else:
                # Extremely unlikely with 50 nodes; skip this distractor
                continue

            t1 = random.uniform(0, self.max_time - self.config.time_window_delta)
            t2 = t1 + random.uniform(0, self.config.time_window_delta)
            events.append((u, v, t1, self._random_feature()))
            events.append((v, w, t2, self._random_feature()))

        # ─── Type 2: correct topology, wrong timing (fires once per sample) ───
        # Uses the actual planted motif times (events[0] and events[1]) to
        # guarantee the distractor A→B and B→C edges cannot pair with any
        # planted edge to form a second within-Δ completion.
        if all(k in nodes_dict for k in ('A', 'B', 'C')) and len(events) >= 2:
            t1_planted = events[0][2]  # planted A→B time
            t2_planted = events[1][2]  # planted B→C time
            delta = self.config.time_window_delta

            # Place distractor A→B at t1_dist such that:
            #   t1_dist cannot pair with planted B→C: |t2_planted - t1_dist| >= Δ
            # Place distractor B→C at t2_dist such that:
            #   t2_dist cannot pair with planted A→B: |t2_dist - t1_planted| >= Δ
            #   t2_dist - t1_dist > Δ (the distractor pair itself is outside Δ)
            # Simplest: put distractor well after the planted motif
            safe_start = t2_planted + delta + 0.1
            if safe_start + delta + 0.1 < self.max_time:
                t1_dist = random.uniform(safe_start, min(safe_start + 10.0, self.max_time - delta - 0.2))
                t2_dist = t1_dist + delta + random.uniform(0.1, 5.0)
                t2_dist = min(t2_dist, self.max_time)
            else:
                # Not enough room after; put distractor well before
                safe_end = t1_planted - delta - 0.1
                if safe_end > delta + 0.2:
                    t2_dist = random.uniform(delta + 0.1, safe_end)
                    t1_dist = t2_dist - delta - random.uniform(0.1, 5.0)
                    t1_dist = max(0.0, t1_dist)
                else:
                    # Fallback: use original logic (retry loop will catch conflicts)
                    t1_dist = random.uniform(0, self.max_time - 2 * delta)
                    t2_dist = t1_dist + delta + random.uniform(0.1, 10.0)
                    t2_dist = min(t2_dist, self.max_time)

            events.append((A, B, t1_dist, self._random_feature()))
            events.append((B, C, t2_dist, self._random_feature()))

    def _add_partial_motifs(self, events: List):
        """Adds incomplete chains (A→B without B→C) — false starts."""
        for _ in range(self.config.num_partial_motifs):
            u, v = random.sample(range(self.config.num_nodes), 2)
            t = random.uniform(0, self.max_time)
            events.append((u, v, t, self._random_feature()))

    def _build_anti_shortcut_context(self, events: List, nodes_dict: Dict):
        """Wraps a motif in noise, distractors, and partial motifs."""
        self._add_noise_events(events, 2 * self.config.noise_multiplier)
        self._add_distractors(events, nodes_dict)
        self._add_partial_motifs(events)

    def _pad_events_to_target(self, events: List):
        """Ensures all samples have exactly the same global event count (e.g. 25)."""
        # Base positive sample size = 2 (motif) + 2*noise + 2*distractors + 2 (type2) + partials
        base_size = 2 + (2 * self.config.noise_multiplier) + (2 * self.config.num_distractors) + 2 + self.config.num_partial_motifs
        # Corruptions can add up to 2 filler edges. We pad to base_size + 4 for safety.
        target_events = base_size + 4
        current = len(events)
        if current < target_events:
            self._add_noise_events(events, target_events - current)
        elif current > target_events:
            raise ValueError(f"Sample generated {current} events, which exceeds the target padding of {target_events}.")

    # ──────────────────────────────────────────────────
    #  Positive sample generation
    # ──────────────────────────────────────────────────
    
    def _dict_to_temporal_data(self, sample_dict: Dict[str, Any]) -> TemporalData:
        """Converts raw event dict to PyG TemporalData format."""
        events = sample_dict['events']
        
        if not events:
            data = TemporalData(
                src=torch.empty(0, dtype=torch.long),
                dst=torch.empty(0, dtype=torch.long),
                t=torch.empty(0, dtype=torch.float),
                msg=torch.empty((0, self.feature_dim), dtype=torch.float)
            )
        else:
            srcs = torch.tensor([e[0] for e in events], dtype=torch.long)
            dsts = torch.tensor([e[1] for e in events], dtype=torch.long)
            times = torch.tensor([e[2] for e in events], dtype=torch.float)
            msgs = torch.stack([e[3] for e in events])
            
            # Resolve exact timestamp collisions (due to max_time clipping or random collisions).
            # Events are already time-sorted from sort_events_by_time(), so srcs/dsts/msgs
            # are aligned with times. We jitter duplicates by +1e-5 which preserves order.
            # NOTE: We do NOT re-sort times independently — that would misalign src/dst/msg.
            while True:
                diffs = times[1:] - times[:-1]
                dup_mask = (diffs == 0)
                if not dup_mask.any():
                    break
                # Use explicit indexing to avoid chained-indexing write-back issues
                dup_indices = torch.where(dup_mask)[0] + 1  # +1 because dup_mask is for times[1:]
                times[dup_indices] += 1e-5

            assert len(times.unique()) == len(times), "Duplicate timestamps found in sample events."

            data = TemporalData(src=srcs, dst=dsts, t=times, msg=msgs)
            
        data.y = torch.tensor([sample_dict['label']], dtype=torch.float)
        
        # We attach motif metadata directly to the TemporalData object
        data.motif_edges = torch.tensor(sample_dict['motif_edges'], dtype=torch.long)
        
        if 'corruption' in sample_dict:
            data.corruption = sample_dict['corruption']
        
        # For batching, we need to store candidate nodes as tensors so PyG can batch them
        A = sample_dict['motif_nodes'].get('A', -1)
        B = sample_dict['motif_nodes'].get('B', -1)
        C = sample_dict['motif_nodes'].get('C', -1)
        data.candidate_nodes = torch.tensor([A, B, C], dtype=torch.long)
        
        # Store source motif timestamps for counterfactual pairing (activation patching)
        if 'source_motif' in sample_dict:
            data.source_motif_t1 = torch.tensor([sample_dict['source_motif']['t1']], dtype=torch.float)
            data.source_motif_t2 = torch.tensor([sample_dict['source_motif']['t2']], dtype=torch.float)
        else:
            data.source_motif_t1 = torch.tensor([-1.0], dtype=torch.float)
            data.source_motif_t2 = torch.tensor([-1.0], dtype=torch.float)

        # Decision step: the event index at which the completion readout snapshots.
        #   Positive samples: the actual B→C event index (motif_edges[1]).
        #   Negative samples: the event index closest to the counterfactual t2
        #     (when B→C would have occurred before corruption).
        # This lets the completion classifier train on both y=0 and y=1 at a
        # comparable temporal decision point.
        if 'decision_step' in sample_dict:
            data.decision_step = torch.tensor([sample_dict['decision_step']], dtype=torch.long)
        else:
            data.decision_step = torch.tensor([-1], dtype=torch.long)

        return data

    def _generate_positive_sample(self, _max_retries: int = 50) -> TemporalData:
        """
        Generates a sample containing the candidate-specific A→B→C motif
        completing within Δ.

        Post-padding validation checks that exactly 1 candidate-specific
        A→B→C completion exists (the planted one). Only the designated
        (A, B, C) triple is checked — arbitrary u→v→w chains elsewhere
        in the graph are irrelevant to the task definition.

        If multiple A→B→C completions are found (e.g. from the type-2
        distractor accidentally landing inside Δ), the sample is
        regenerated. Raises RuntimeError if all retries fail.
        """
        for _attempt in range(_max_retries):
            A, B, C = random.sample(range(self.config.num_nodes), 3)
            nodes_dict = {'A': A, 'B': B, 'C': C}

            t1 = random.uniform(0, self.max_time - self.config.time_window_delta)
            t2 = t1 + random.uniform(0.01, self.config.time_window_delta)  # strictly positive gap

            events = [
                (A, B, t1, self._random_feature()),  # edge 1
                (B, C, t2, self._random_feature()),  # edge 2
            ]

            self._build_anti_shortcut_context(events, nodes_dict)
            self._pad_events_to_target(events)
            sorted_events = sort_events_by_time(events)

            # Candidate-specific validation: exactly 1 A→B→C completion within Δ
            n_candidate = _count_candidate_motifs(
                sorted_events, A, B, C, self.config.time_window_delta)
            if n_candidate != 1:
                continue  # retry — multiple A→B→C completions from distractor overlap

            # Find motif edge indices after sorting
            motif_edges = []
            for i, ev in enumerate(sorted_events):
                if ev[0] == A and ev[1] == B and ev[2] == t1:
                    motif_edges.append(i)
                elif ev[0] == B and ev[1] == C and ev[2] == t2:
                    motif_edges.append(i)

            # Decision step for completion readout = the B→C event index
            decision_step = motif_edges[1] if len(motif_edges) >= 2 else -1

            sample_dict = {
                'events': sorted_events,
                'label': 1,
                'motif_edges': motif_edges,
                'motif_nodes': nodes_dict,
                'corruption': 'none',
                'decision_step': decision_step,
            }
            return self._dict_to_temporal_data(sample_dict)

        raise RuntimeError(
            f"Positive sample generation: could not produce exactly 1 "
            f"candidate-specific A→B→C motif after {_max_retries} attempts. "
            f"This indicates a configuration issue (e.g. type-2 distractor "
            f"timing overlaps with planted motif). Consider widening "
            f"time_window_delta or adjusting distractor generation."
        )

    # ──────────────────────────────────────────────────
    #  Negative sample generation (inline corruptions)
    # ──────────────────────────────────────────────────
    
    def _generate_negative_sample(self, _depth: int = 0) -> TemporalData:
        """
        Generates a negative sample by creating a valid positive sample
        and then applying ONE minimal corruption to break the motif.

        The corruption is applied BEFORE adding noise/distractors so that
        the positive and negative samples are structurally identical except
        for the single corrupted element.

        Post-generation validation confirms no accidental A→B→C completion
        exists for the candidate triple. If one is found, the entire sample
        is regenerated (not patched) to preserve corruption semantics.
        """
        if _depth > 50:
            raise RuntimeError(
                "Negative sample generation: could not produce a sample without "
                "an accidental candidate motif after 50 attempts."
            )
        A, B, C = random.sample(range(self.config.num_nodes), 3)
        nodes_dict = {'A': A, 'B': B, 'C': C}
        
        t1 = random.uniform(0, self.max_time - self.config.time_window_delta)
        t2 = t1 + random.uniform(0.01, self.config.time_window_delta)
        
        # Save pre-corruption motif parameters for counterfactual pairing
        source_motif = {'A': A, 'B': B, 'C': C, 't1': t1, 't2': t2}
        
        # Start with valid motif edges then corrupt one
        strategy = random.choice(self.corruption_strategies)
        
        if strategy == 'topology_break':
            # ── Strategy A: break shared-node identity ──
            # Change B→C to D→C where D ≠ B
            D = self._pick_other_node(exclude={A, B, C})
            events = [
                (A, B, t1, self._random_feature()),
                (D, C, t2, self._random_feature()),   # D replaces B as source
            ]
            
        elif strategy == 'timing_break':
            # ── Strategy B: push completing edge outside time window ──
            t2_corrupt = min(self.max_time, t1 + self.config.time_window_delta + random.uniform(0.1, 5.0))
            events = [
                (A, B, t1, self._random_feature()),
                (B, C, t2_corrupt, self._random_feature()),
            ]
            
        elif strategy == 'edge_deletion':
            # ── Strategy C: delete the completing edge entirely ──
            # Replace it with a random noise edge (no self-loops)
            u = random.randint(0, self.config.num_nodes - 1)
            v = random.randint(0, self.config.num_nodes - 1)
            while u == v:
                v = random.randint(0, self.config.num_nodes - 1)
            events = [
                (A, B, t1, self._random_feature()),
                (u, v, t2, self._random_feature()),   # random edge, not B→C
            ]
            
        elif strategy == 'source_shuffle':
            # ── Strategy D: change chain origin ──
            # Change A→B to X→B where X ≠ A
            X = self._pick_other_node(exclude={A, B, C})
            events = [
                (X, B, t1, self._random_feature()),   # X replaces A
                (B, C, t2, self._random_feature()),
            ]
            
        elif strategy == 'temporal_misordering':
            # ── Strategy E: swap timestamps ──
            # t₂ comes before t₁ → violates temporal ordering
            events = [
                (A, B, t2, self._random_feature()),   # swapped: uses t2
                (B, C, t1, self._random_feature()),   # swapped: uses t1
            ]
        
        # ── Safeguard: Prevent degree-based shortcuts ──
        # Enforce base motif degree parity BEFORE adding noise/distractors.
        # In a positive sample, A has degree 1, B has degree 2, C has degree 1.
        # Under corruptions, some of these may drop. We inject compensating 
        # filler edges to restore the exact base motif degree.
        motif_node_set = {A, B, C}
        
        a_degree = sum(1 for e in events if e[0] == A or e[1] == A)
        if a_degree < 1:
            other = self._pick_other_node(exclude=motif_node_set)
            t_filler = random.uniform(0, self.max_time)
            events.append((A, other, t_filler, self._random_feature()) if random.random() < 0.5 else (other, A, t_filler, self._random_feature()))
            
        b_degree = sum(1 for e in events if e[0] == B or e[1] == B)
        while b_degree < 2:
            other = self._pick_other_node(exclude=motif_node_set)
            t_filler = random.uniform(0, self.max_time)
            events.append((B, other, t_filler, self._random_feature()) if random.random() < 0.5 else (other, B, t_filler, self._random_feature()))
            b_degree += 1
            
        c_degree = sum(1 for e in events if e[0] == C or e[1] == C)
        if c_degree < 1:
            other = self._pick_other_node(exclude=motif_node_set)
            t_filler = random.uniform(0, self.max_time)
            events.append((C, other, t_filler, self._random_feature()) if random.random() < 0.5 else (other, C, t_filler, self._random_feature()))
            
        self._build_anti_shortcut_context(events, nodes_dict)
        self._pad_events_to_target(events)
        sorted_events = sort_events_by_time(events)

        # ── Post-padding candidate-specific motif check ──
        # Verify that no valid A→B→C completion exists for the designated
        # candidate triple. Only the candidate nodes matter — arbitrary
        # u→v→w chains elsewhere are irrelevant to the classification task.
        # If an accidental A→B→C completion is found (e.g. from noise edges
        # that happen to connect A→B or B→C within Δ), regenerate entirely
        # to avoid corrupting the intended corruption semantics.
        if _has_candidate_motif(sorted_events, A, B, C,
                                self.config.time_window_delta):
            # Rare: noise/distractor accidentally completed A→B→C.
            # Regenerate from scratch to preserve corruption semantics.
            return self._generate_negative_sample(_depth=_depth + 1)

        # Counterfactual decision step: the last event with t ≤ t2.
        # For positives, this is exactly the B→C event.
        # For negatives, this is the model's memory state at the
        # corresponding temporal boundary — the last real event
        # processed before the counterfactual completion time.
        # This ensures both classes are evaluated at precisely
        # the same temporal decision boundary.
        counterfactual_t2 = source_motif['t2']
        decision_step = -1
        for i, ev in enumerate(sorted_events):
            if ev[2] <= counterfactual_t2:
                decision_step = i
            else:
                break

        sample_dict = {
            'events': sorted_events,
            'label': 0,
            'motif_edges': [],  # no valid motif edges in negative samples
            'motif_nodes': nodes_dict,
            'corruption': strategy,
            'source_motif': source_motif,
            'decision_step': decision_step,
        }
        return self._dict_to_temporal_data(sample_dict)

    # ──────────────────────────────────────────────────
    #  Dataset interface
    # ──────────────────────────────────────────────────
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @staticmethod
    def collate_fn(batch):
        """Returns batch as a PyG Batch of TemporalData objects."""
        return Batch.from_data_list(batch)
