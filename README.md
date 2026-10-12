# Reverse-Engineering a Temporal Graph Network

A mechanistic interpretability (MI) study of a Temporal Graph Network (TGN) with recurrent per-node memory, run on a synthetic task whose correct circuit is known in advance.

Most MI work targets transformers, where attention gives a conveniently traceable computation graph. A TGN is a different substrate: it carries per-node memory states that persist across time steps and are rewritten by GRU gates as events arrive, so information written into node `A`'s memory can be read dozens of events later. This codebase trains such a model and then takes it apart with activation patching, gate clamping, and checkpoint-by-checkpoint replay of how the circuit forms over training.

Working synthetically is the central methodological commitment. Running MI on a model trained for some real-world objective produces mechanistic observations with nothing to check them against; designing the task means the correct computational strategy is specified up front, which reduces the study to a single question — did the model find it?

---

## Layout

```
temporal_gnn_mi/
  config.py                     Frozen dataclass configs
  models/   tgn.py              Main TGN, 15 intervention points
            memory.py           Per-node memory, hand-rolled GRU
            message.py          Message function + aggregator
            time_encoding.py    Learnable sinusoidal encoding
            embedding.py        Identity embedding
            flat_gru.py         Ablation baseline (no per-node memory)
  datasets/ temporal_motif_generator.py   Synthetic generator + anti-shortcut logic
  training/ trainer.py          Training loop, dual checkpointing

experiments/
  static/     exp1-exp8         Static MI experiments
  dynamics/   expT1-expT5       Circuit formation over training
  utils.py                      Model loading, checkpoint discovery, plotting

scripts/
  run_all_seeds.py              Master pipeline (train → audit → experiments → baseline → compile)
  run_experiments.py            Experiment registry and dispatcher
  train_tgn.py                  Standalone training
  audit_model.py                Shortcut/leakage audit
  compile_results.py            Cross-seed aggregation
```

## Install

```bash
pip install -r requirements.txt
```

Needs PyTorch ≥ 2.0. A GPU is strongly recommended — training is ~178 s/epoch on a Tesla T4, so a full four-seed run takes roughly a day.

## Running

Everything, all four seeds:

```bash
python scripts/run_all_seeds.py --seeds 42 43 44 45 --completion_readout
```

> **Pass `--completion_readout` to reproduce the published numbers.** It defaults to off, and without it the second classifier head is never trained, so the completion-readout results will be missing.

Useful flags: `--experiments_only` (skip training, reuse checkpoints), `--compile_only` (cross-seed aggregation only), `--skip_exp7` (skip the 250-permutation null control, the most expensive step), `--skip_flatgru`, `--base_dir`.

Single seed, step by step:

```bash
python scripts/train_tgn.py --seed 42 --output_dir runs/seed_42
python scripts/audit_model.py runs/seed_42/best_model.pt
python scripts/run_experiments.py \
    --model_path runs/seed_42/best_model.pt \
    --checkpoint_dir runs/seed_42 \
    --output_dir runs/seed_42/experiments --seed 42
python scripts/compile_results.py --base_dir runs --seeds 42 43 44 45
```

Selected experiments:

```bash
python scripts/run_experiments.py --list                       # show registry
python scripts/run_experiments.py --experiments exp2 exp3b T1   # specific
python scripts/run_experiments.py --category dynamics           # T1-T5 only
```

| Key | Experiment | Key | Experiment |
|---|---|---|---|
| `exp1` | Causal importance map | `exp7` | Shuffled-label null control |
| `exp2` | Composition test | `exp8` | Memory knockout |
| `exp3` | Observational gate analysis | `T1` | Circuit formation |
| `exp3b` | Causal gate interventions | `T2` | Corruption learning order |
| `exp4` | Time-window detection | `T3` | Gate specialization |
| `exp5` | Source-shuffle failure analysis | `T4` | Time-boundary sharpness |
| `exp6` | Ablation method validation | `T5` | Frequency evolution |

That is 13 experiments, 8 static and 5 dynamics. `exp9_cross_seed_consistency.py` is an aggregation helper, not one of the 13, and is not in the dispatcher registry.

### Checkpoints and output

The dynamics experiments replay the training trajectory and so need epoch checkpoints, not just `best_model.pt` — pass `--checkpoint_dir`. The trainer writes full checkpoints (model + optimizer) every 5 epochs, and lightweight analysis checkpoints both adaptively, on >5% validation improvement, and at fixed milestones `{1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200}`. The milestones exist so that epoch 10 in one seed can be compared directly against epoch 10 in another; the adaptive saves cluster wherever the model is changing fastest.

Output lands in `runs/seed_{N}/`:

```
best_model.pt                      Early-stopping best
checkpoint_ep{N}.pt                Full checkpoints, every 5 epochs
analysis_checkpoints/epoch_{N}.pt  Adaptive + milestone, for T1-T5
audit_results.json                 Shortcut audit
experiments/                       Raw .pt results + .png figures per experiment
```

## Citation

Anupam Dwivedi, *Reverse-Engineering a Temporal Graph Network: A Complete Technical Report on Mechanistic Interpretability of Recurrent Per-Node Memory* (2026).
