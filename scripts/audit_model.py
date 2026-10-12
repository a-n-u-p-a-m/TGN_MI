"""
Rigorous Audit Script for Trained TGN Models.
Tests whether the trained TGN has learned shortcuts by training
trivial baseline MLPs to predict the TGN's outputs.

6 baselines tested:
  A) Feature-only     — mean edge features
  B) Degree-only      — candidate node degree counts
  C) Event-count      — total number of events
  D) Timestamp-only   — full sorted timestamp vector
  E) Node-identity    — raw candidate node IDs
  F) Candidate-timing — timing stats of candidate-touching events

Usage: python scripts/audit_model.py <path_to_best_model.pt>
"""
import sys
import os
import json
import torch
import torch.nn as nn
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporal_gnn_mi.config import DatasetConfig, ModelConfig
from temporal_gnn_mi.datasets.temporal_motif_generator import TemporalMotifDataset
from temporal_gnn_mi.datasets.batch_utils import temporal_collate_fn
from temporal_gnn_mi.models.tgn import TGN

def run_audit(checkpoint_path: str):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load optimal hyperparameters
    config_path = os.path.join(os.path.dirname(checkpoint_path), 'training_config.json')
    config = ModelConfig()
    num_nodes = 50  # Default verified from training run
    
    # ── Read noise_multiplier from config (not hardcoded) ──
    noise_mult = 4  # default fallback
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = json.load(f)
            model_cfg = cfg.get('model', {})
            config.memory_dim = model_cfg.get('memory_dim', config.memory_dim)
            config.message_dim = model_cfg.get('message_dim', config.message_dim)
            config.time_dim = model_cfg.get('time_dim', config.time_dim)
            config.embedding_dim = model_cfg.get('embedding_dim', config.embedding_dim)
            config.dropout = model_cfg.get('dropout', config.dropout)
            # Try to infer num_nodes if saved, otherwise keep 50
            dataset_cfg = cfg.get('dataset', {})
            num_nodes = dataset_cfg.get('num_nodes', 50)
            # Read noise_multiplier from config with fallback chain
            noise_mult = dataset_cfg.get('noise_multiplier',
                         model_cfg.get('noise_multiplier', 7))
            
    print(f"Model config initialized. Expected nodes: {num_nodes}, noise_multiplier: {noise_mult}")
    
    # Initialize model
    model = TGN(config, num_nodes=num_nodes, edge_feat_dim=16)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Generate an evaluation dataset with MATCHING noise_multiplier
    total_samples = 2500
    print(f"Generating rigorous evaluation dataset ({total_samples} samples at {num_nodes} nodes, noise_mult={noise_mult})...")
    dataset_config = DatasetConfig(num_nodes=num_nodes, seed=101, noise_multiplier=noise_mult)
    dataset = TemporalMotifDataset(dataset_config, num_samples=total_samples, split='test')
    
    # Extract features and model predictions
    print("Extracting model predictions and baseline features...")
    feat_X, degree_X, count_X = [], [], []
    time_X, node_id_X, pairwise_X = [], [], []
    tgn_preds = []
    corruptions = []
    
    with torch.no_grad():
        for s in dataset.samples:
            corruptions.append(getattr(s, 'corruption', 'none'))
            
            # Baseline A: mean edge features
            if s.msg.size(0) > 0:
                feat_X.append(s.msg.mean(dim=0))
            else:
                feat_X.append(torch.zeros(16))
            
            # Baseline B: degree counts for A, B, C
            A, B, C = s.candidate_nodes.tolist()
            all_nodes = torch.cat([s.src, s.dst]).tolist()
            deg_a = all_nodes.count(A)
            deg_b = all_nodes.count(B)
            deg_c = all_nodes.count(C)
            degree_X.append(torch.tensor([deg_a, deg_b, deg_c], dtype=torch.float))
            
            # Baseline C: total event count
            count_X.append(torch.tensor([s.src.size(0)], dtype=torch.float))
            
            # Baseline D: full sorted timestamp vector
            time_X.append(s.t.clone())  # (max_events,)
            
            # Baseline E: raw candidate node IDs
            node_id_X.append(s.candidate_nodes.float())  # (3,)
            
            # Baseline F: timing features of candidate-touching events
            cand_set = {A, B, C}
            cand_times = []
            for j in range(s.src.size(0)):
                if s.src[j].item() in cand_set or s.dst[j].item() in cand_set:
                    cand_times.append(s.t[j].item())
            
            if len(cand_times) >= 2:
                cand_times.sort()
                gaps = [cand_times[i+1] - cand_times[i] for i in range(len(cand_times)-1)]
                min_gap = min(gaps)
                mean_gap = sum(gaps) / len(gaps)
                num_cand_events = len(cand_times)
            else:
                min_gap = 100.0
                mean_gap = 100.0
                num_cand_events = len(cand_times)
            
            pairwise_X.append(torch.tensor([min_gap, mean_gap, float(num_cand_events)], dtype=torch.float))
            
            # TGN prediction
            batch = temporal_collate_fn([s])
            logit = model(batch).item()
            pred = 1.0 if logit > 0 else 0.0
            tgn_preds.append(pred)
            
            if hasattr(model, 'clear_hooks'):
                model.clear_hooks()
                
    feat_X = torch.stack(feat_X)
    degree_X = torch.stack(degree_X)
    count_X = torch.stack(count_X)
    time_X = torch.stack(time_X)       # (N, max_events)
    node_id_X = torch.stack(node_id_X) # (N, 3)
    pairwise_X = torch.stack(pairwise_X) # (N, 3)
    tgn_preds = torch.tensor(tgn_preds, dtype=torch.float).unsqueeze(1)

    # Ground-truth labels for dual audit
    y_true = torch.tensor([s.y.item() for s in dataset.samples], dtype=torch.float).unsqueeze(1)

    # Train/Test Split (80/20)
    train_size = int(0.8 * total_samples)

    X_train_dict = {
        'Feature-only':     feat_X[:train_size],
        'Degree-only':      degree_X[:train_size],
        'Event-count':      count_X[:train_size],
        'Timestamp-only':   time_X[:train_size],
        'Node-identity':    node_id_X[:train_size],
        'Candidate-timing': pairwise_X[:train_size],
    }
    X_test_dict = {
        'Feature-only':     feat_X[train_size:],
        'Degree-only':      degree_X[train_size:],
        'Event-count':      count_X[train_size:],
        'Timestamp-only':   time_X[train_size:],
        'Node-identity':    node_id_X[train_size:],
        'Candidate-timing': pairwise_X[train_size:],
    }
    test_corruptions = corruptions[train_size:]

    dims = {
        'Feature-only':     16,
        'Degree-only':      3,
        'Event-count':      1,
        'Timestamp-only':   time_X.size(1),  # max_events
        'Node-identity':    3,
        'Candidate-timing': 3,
    }

    all_passed = True
    audit_results = {}

    # ── Dual audit: predict both TGN predictions AND ground-truth labels ──
    for target_name, y_full in [("TGN predictions (y_hat)", tgn_preds),
                                 ("Ground-truth labels (y_true)", y_true)]:
        y_train = y_full[:train_size]
        y_test = y_full[train_size:]

        print(f"\n{'─'*50}")
        print(f"  AUDIT TARGET: {target_name}")
        print(f"  Training baselines (N={train_size}), evaluating (N={total_samples - train_size})")
        print(f"{'─'*50}")

        for name in dims:
            in_dim = dims[name]
            X_train = X_train_dict[name]
            X_test = X_test_dict[name]

            # Timestamp-only gets a larger hidden layer due to high input dimensionality
            if name == 'Timestamp-only':
                mlp = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 1))
            else:
                mlp = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 1))

            opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()

            # Train to predict target
            for _ in range(150):
                opt.zero_grad()
                loss = criterion(mlp(X_train), y_train)
                loss.backward()
                opt.step()

            # Evaluate on Test Set
            with torch.no_grad():
                preds = (torch.sigmoid(mlp(X_test)) > 0.5).float()
                overall_acc = (preds == y_test).float().mean().item()

                # Compute balanced accuracy
                pos_mask = (y_test == 1.0)
                neg_mask = (y_test == 0.0)
                pos_acc = (preds[pos_mask] == y_test[pos_mask]).float().mean().item() if pos_mask.any() else 0.0
                neg_acc = (preds[neg_mask] == y_test[neg_mask]).float().mean().item() if neg_mask.any() else 0.0
                balanced_acc = (pos_acc + neg_acc) / 2.0

                # Per-corruption analysis
                corr_correct = defaultdict(int)
                corr_total = defaultdict(int)

                for i, corr in enumerate(test_corruptions):
                    corr_total[corr] += 1
                    if preds[i].item() == y_test[i].item():
                        corr_correct[corr] += 1

            # A baseline only counts as a shortcut if it can predict BOTH classes better than chance.
            status = "WARN (Shortcut!)" if balanced_acc >= 0.65 else "OK"
            if balanced_acc >= 0.65:
                all_passed = False

            print(f"\n[{name}] balanced test accuracy: {balanced_acc:.2%} {status}")
            print(f"    (Overall: {overall_acc:.2%}, Pos: {pos_acc:.2%}, Neg: {neg_acc:.2%})")

            # Check if any specific corruption leaks
            for corr in sorted(corr_total.keys()):
                acc = corr_correct[corr] / corr_total[corr]
                tag = "[WARN]" if (acc >= 0.65 and balanced_acc >= 0.65) else "[OK]"
                print(f"    - {corr:22s} accuracy: {acc:5.1%} {tag}")

            audit_results[f"{target_name}|{name}"] = balanced_acc

    print("\n" + "="*50)
    print("CONCLUSION:")
    if all_passed:
        print("  OK! No baseline can predict TGN outputs OR ground-truth labels")
        print("  from simple features. No shortcut detected among the evaluated baseline feature families.")
    else:
        print("  WARNING: A baseline accurately predicted outputs (balanced acc >= 65%).")
        print("  Possible shortcut detected — check which target and baseline above.")

    # Save audit results as JSON
    audit_json = {
        'passed': all_passed,
        'results': {k: float(v) for k, v in audit_results.items()},
    }
    audit_path = os.path.join(os.path.dirname(checkpoint_path), 'audit_results.json')
    with open(audit_path, 'w') as f:
        json.dump(audit_json, f, indent=2)
    print(f"  Saved: {audit_path}")

    return all_passed

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/audit_model.py <checkpoint_path>")
        sys.exit(1)
    run_audit(sys.argv[1])
