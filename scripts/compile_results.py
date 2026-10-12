"""
Cross-Seed Result Compiler

Aggregates experiment results across multiple seeds and computes
summary statistics (mean ± std) for all key metrics.

Usage:
    python -m scripts.compile_results
    python -m scripts.compile_results --base_dir runs --seeds 42 43 44 45
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _safe_load_pt(path):
    """Load a .pt file, returning None if missing."""
    if not os.path.exists(path):
        return None
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


def _safe_load_json(path):
    """Load a JSON file, returning None if missing."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def compile_training_metrics(base_dir, seeds):
    """Aggregate training metrics across seeds."""
    metrics = {'roc_auc': [], 'accuracy': [], 'ap': []}

    for seed in seeds:
        run_dir = os.path.join(base_dir, f'seed_{seed}')
        found = False

        # Try JSON files first
        for fname in ['best_metrics.json', 'training_metrics.json']:
            data = _safe_load_json(os.path.join(run_dir, fname))
            if data:
                for k in metrics:
                    if k in data:
                        metrics[k].append(data[k])
                found = True
                break

        # Fallback: extract from best_model.pt checkpoint
        if not found:
            ckpt = _safe_load_pt(os.path.join(run_dir, 'best_model.pt'))
            if ckpt and 'metrics' in ckpt:
                data = ckpt['metrics']
                for k in metrics:
                    if k in data:
                        metrics[k].append(data[k])
                found = True

        # Fallback: parse training_history.json for best val metrics
        if not found:
            hist = _safe_load_json(os.path.join(run_dir, 'training_history.json'))
            if hist and 'val_roc_auc' in hist:
                best_idx = int(np.argmax(hist['val_roc_auc']))
                metrics['roc_auc'].append(hist['val_roc_auc'][best_idx])
                if 'val_accuracy' in hist:
                    metrics['accuracy'].append(hist['val_accuracy'][best_idx])
                if 'val_ap' in hist:
                    metrics['ap'].append(hist['val_ap'][best_idx])

    result = {}
    for k, vals in metrics.items():
        if vals:
            result[k] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'values': vals,
                'n': len(vals),
            }
    return result


def compile_flatgru_metrics(base_dir, seeds):
    """Aggregate Flat-GRU baseline metrics across seeds."""
    metrics = {'roc_auc': [], 'accuracy': [], 'ap': []}

    for seed in seeds:
        flat_dir = os.path.join(base_dir, f'seed_{seed}', 'flat_gru')
        found = False

        for fname in ['best_metrics.json', 'training_metrics.json']:
            data = _safe_load_json(os.path.join(flat_dir, fname))
            if data:
                for k in metrics:
                    if k in data:
                        metrics[k].append(data[k])
                found = True
                break

        if not found:
            ckpt = _safe_load_pt(os.path.join(flat_dir, 'best_model.pt'))
            if ckpt and 'metrics' in ckpt:
                data = ckpt['metrics']
                for k in metrics:
                    if k in data:
                        metrics[k].append(data[k])
                found = True

        if not found:
            hist = _safe_load_json(os.path.join(flat_dir, 'training_history.json'))
            if hist and 'val_roc_auc' in hist:
                best_idx = int(np.argmax(hist['val_roc_auc']))
                metrics['roc_auc'].append(hist['val_roc_auc'][best_idx])
                if 'val_accuracy' in hist:
                    metrics['accuracy'].append(hist['val_accuracy'][best_idx])
                if 'val_ap' in hist:
                    metrics['ap'].append(hist['val_ap'][best_idx])

    result = {}
    for k, vals in metrics.items():
        if vals:
            result[k] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'values': vals,
                'n': len(vals),
            }
    return result


def compile_exp1_causal_map(base_dir, seeds):
    """Aggregate Experiment 1 (Causal Importance Map) results."""
    all_maps = []
    for seed in seeds:
        path = os.path.join(base_dir, f'seed_{seed}', 'experiments',
                            'experiment_1_causal_map', 'experiment_1_causal_map.pt')
        data = _safe_load_pt(path)
        if data:
            # Support both key names for backward compatibility
            key = 'mean_effects' if 'mean_effects' in data else 'importance_map'
            if key in data:
                all_maps.append(data[key].numpy())

    if not all_maps:
        return None

    stacked = np.stack(all_maps)
    return {
        'mean_map': stacked.mean(axis=0).tolist(),
        'std_map': stacked.std(axis=0).tolist(),
        'n_seeds': len(all_maps),
    }


def compile_exp2_composition(base_dir, seeds):
    """Aggregate Experiment 2 (Composition Recovery Ratio) results.

    Primary statistic: ratio_of_means (more stable than mean_of_ratios).
    Also aggregates mean_raw_gain and bootstrap CI when available.
    Falls back to mean_ratio for backward compatibility with older results.
    """
    ratio_of_means_list = []
    mean_raw_gain_list = []
    ci95_lower_list = []
    ci95_upper_list = []
    # Fallback: mean_ratio (less stable, kept for backward compat)
    mean_ratio_list = []
    stds = []
    n_valid = []

    for seed in seeds:
        path = os.path.join(base_dir, f'seed_{seed}', 'experiments',
                            'experiment_2_composition', 'experiment_2_composition.pt')
        data = _safe_load_pt(path)
        if data is None:
            continue

        # Primary: ratio_of_means
        if 'ratio_of_means' in data:
            ratio_of_means_list.append(float(data['ratio_of_means']))
        # Secondary: mean_raw_gain
        if 'mean_raw_gain' in data:
            mean_raw_gain_list.append(float(data['mean_raw_gain']))
        # Bootstrap CI
        if 'ratio_of_means_ci95' in data:
            ci = data['ratio_of_means_ci95']
            if isinstance(ci, (list, tuple)) and len(ci) == 2:
                ci95_lower_list.append(float(ci[0]))
                ci95_upper_list.append(float(ci[1]))
        # Fallback: mean_ratio
        if 'mean_ratio' in data:
            mean_ratio_list.append(float(data['mean_ratio']))
        if 'std_ratio' in data:
            stds.append(float(data['std_ratio']))
        if 'num_valid_samples' in data:
            n_valid.append(int(data['num_valid_samples']))

    # Need at least one seed with data
    if not ratio_of_means_list and not mean_ratio_list:
        return None

    result = {}

    # Use ratio_of_means as the primary recovery_ratio if available
    if ratio_of_means_list:
        result['recovery_ratio'] = {
            'mean': float(np.mean(ratio_of_means_list)),
            'std': float(np.std(ratio_of_means_list)),
            'per_seed': ratio_of_means_list,
            'statistic': 'ratio_of_means',
        }
    elif mean_ratio_list:
        # Fallback for older results that only have mean_ratio
        result['recovery_ratio'] = {
            'mean': float(np.mean(mean_ratio_list)),
            'std': float(np.std(mean_ratio_list)),
            'per_seed': mean_ratio_list,
            'statistic': 'mean_ratio (fallback)',
        }

    if mean_raw_gain_list:
        result['mean_raw_gain'] = {
            'mean': float(np.mean(mean_raw_gain_list)),
            'std': float(np.std(mean_raw_gain_list)),
            'per_seed': mean_raw_gain_list,
        }

    if ci95_lower_list and ci95_upper_list:
        result['ratio_of_means_ci95'] = {
            'lower': ci95_lower_list,
            'upper': ci95_upper_list,
        }

    if stds:
        result['within_seed_std'] = {
            'mean': float(np.mean(stds)),
            'values': stds,
        }

    if n_valid:
        result['valid_samples'] = {
            'mean': float(np.mean(n_valid)),
            'values': n_valid,
        }

    result['n_seeds'] = max(len(ratio_of_means_list), len(mean_ratio_list))

    return result


def compile_exp2_permutation(base_dir, seeds):
    """Aggregate Experiment 2 permutation test results."""
    p_values = []
    observed_means = []
    ci_lowers = []
    ci_uppers = []

    for seed in seeds:
        path = os.path.join(base_dir, f'seed_{seed}', 'experiments',
                            'experiment_2_composition', 'experiment_2_permutation_test.pt')
        data = _safe_load_pt(path)
        if data:
            p_values.append(float(data['p_value']))
            observed_means.append(float(data['observed_mean']))
            ci_lowers.append(float(data['ci_95_lower']))
            ci_uppers.append(float(data['ci_95_upper']))

    if not p_values:
        return None

    return {
        'p_values': p_values,
        'mean_p_value': float(np.mean(p_values)),
        'all_significant_005': all(p < 0.05 for p in p_values),
        'all_significant_001': all(p < 0.001 for p in p_values),
        'observed_means': observed_means,
        'ci_95': {
            'lower': ci_lowers,
            'upper': ci_uppers,
        },
        'n_seeds': len(p_values),
    }


def compile_exp3_gate_analysis(base_dir, seeds):
    """Aggregate Experiment 3A (Gate-Level Analysis) results."""
    all_d_prime_z = []
    all_d_prime_r = []

    for seed in seeds:
        # Try both directory naming conventions
        for dirname, fname in [
            ('experiment_3a_gate_analysis', 'experiment_3a_gate_analysis.pt'),
            ('experiment_3_gate_analysis', 'experiment_3_gate_analysis.pt'),
        ]:
            path = os.path.join(base_dir, f'seed_{seed}', 'experiments', dirname, fname)
            data = _safe_load_pt(path)
            if data:
                # Support both key names
                z_key = 'z_discriminability' if 'z_discriminability' in data else 'dprime_z'
                r_key = 'r_discriminability' if 'r_discriminability' in data else 'dprime_r'
                if z_key in data:
                    all_d_prime_z.append(data[z_key].numpy())
                if r_key in data:
                    all_d_prime_r.append(data[r_key].numpy())
                break

    if not all_d_prime_z and not all_d_prime_r:
        return None

    result = {'n_seeds': max(len(all_d_prime_z), len(all_d_prime_r))}

    if all_d_prime_z:
        stacked = np.stack(all_d_prime_z)
        result['dprime_z'] = {
            'mean_per_dim': stacked.mean(axis=0).tolist(),
            'mean_overall': float(stacked.mean()),
            'std_overall': float(stacked.std()),
            'max_per_seed': [float(x.max()) for x in all_d_prime_z],
        }

    if all_d_prime_r:
        stacked = np.stack(all_d_prime_r)
        result['dprime_r'] = {
            'mean_per_dim': stacked.mean(axis=0).tolist(),
            'mean_overall': float(stacked.mean()),
            'std_overall': float(stacked.std()),
            'max_per_seed': [float(x.max()) for x in all_d_prime_r],
        }

    return result


def compile_exp3b_causal_gate(base_dir, seeds):
    """Aggregate Experiment 3B (Causal Gate Clamping) results."""
    effects = {'clamp_z': [], 'clamp_r': [], 'clamp_both': [], 'shuffle_z': []}

    for seed in seeds:
        # Try both naming conventions
        for dirname, fname in [
            ('experiment_3b_causal_gate', 'experiment_3b_causal_gate.pt'),
            ('experiment_3b_causal_gates', 'experiment_3b_causal_gates.pt'),
        ]:
            path = os.path.join(base_dir, f'seed_{seed}', 'experiments', dirname, fname)
            data = _safe_load_pt(path)
            if data and 'effects' in data:
                for key in effects:
                    if key in data['effects']:
                        eff = data['effects'][key]
                        if 'mean_diff' in eff:
                            effects[key].append(float(eff['mean_diff']))
                break  # Found data, stop trying naming variants

    if not any(effects.values()):
        return None

    result = {'n_seeds': max(len(v) for v in effects.values())}
    for key, vals in effects.items():
        if vals:
            result[key] = {
                'mean_effect': float(np.mean(vals)),
                'std_effect': float(np.std(vals)),
                'per_seed': vals,
            }
    return result


def compile_exp4_time_window(base_dir, seeds):
    """Aggregate Experiment 4 (Time-Window Mechanism) results."""
    transition_points = []

    for seed in seeds:
        path = os.path.join(base_dir, f'seed_{seed}', 'experiments',
                            'experiment_4_time_window', 'experiment_4_time_window.pt')
        data = _safe_load_pt(path)
        if data:
            # Look for transition point (where logit crosses 0)
            # Support both key names
            dt_key = 'delta_t_sweep' if 'delta_t_sweep' in data else 'delta_ts'
            logit_key = 'logit_matrix' if 'logit_matrix' in data else 'logits'
            if dt_key in data and logit_key in data:
                dts = data[dt_key].numpy()
                logits_raw = data[logit_key].numpy()
                # logit_matrix is (samples, sweep_points) — take mean across samples
                logits = logits_raw.mean(axis=0) if logits_raw.ndim == 2 else logits_raw
                # Find zero crossing
                for j in range(len(logits) - 1):
                    if logits[j] > 0 and logits[j + 1] <= 0:
                        # Linear interpolation
                        frac = logits[j] / (logits[j] - logits[j + 1])
                        tp = dts[j] + frac * (dts[j + 1] - dts[j])
                        transition_points.append(float(tp))
                        break

    if not transition_points:
        return None

    return {
        'transition_point': {
            'mean': float(np.mean(transition_points)),
            'std': float(np.std(transition_points)),
            'per_seed': transition_points,
        },
        'n_seeds': len(transition_points),
    }


def compile_exp5_source_shuffle(base_dir, seeds):
    """Aggregate Experiment 5 (Source-Shuffle Failure) results."""
    cosine_sims = []

    for seed in seeds:
        path = os.path.join(base_dir, f'seed_{seed}', 'experiments',
                            'experiment_5_source_shuffle', 'experiment_5_source_shuffle.pt')
        data = _safe_load_pt(path)
        if data:
            if 'mean_cosine_sim' in data:
                cosine_sims.append(float(data['mean_cosine_sim']))
            elif 'cos_sims' in data:
                # cos_sims is nested: {group: {node: tensor}}
                # Compute overall mean across all groups and nodes
                all_means = []
                for group, nodes in data['cos_sims'].items():
                    for node, vals in nodes.items():
                        all_means.append(float(vals.mean()))
                if all_means:
                    cosine_sims.append(float(np.mean(all_means)))

    if not cosine_sims:
        return None

    return {
        'mean_cosine_similarity': float(np.mean(cosine_sims)),
        'std_cosine_similarity': float(np.std(cosine_sims)),
        'per_seed': cosine_sims,
        'n_seeds': len(cosine_sims),
    }


def compile_exp6_ablation(base_dir, seeds):
    """Aggregate Experiment 6 (Ablation Comparison) results."""
    correlations = []
    zero_rel_vars = []
    mean_rel_vars = []

    for seed in seeds:
        path = os.path.join(base_dir, f'seed_{seed}', 'experiments',
                            'experiment_6_ablation_comparison',
                            'experiment_6_ablation_comparison.pt')
        data = _safe_load_pt(path)
        if data and 'correlation' in data:
            correlations.append(float(data['correlation']))
            # Support both old 'zero_snr' and new 'zero_rel_var' keys
            if 'zero_rel_var' in data:
                zero_rel_vars.append(float(data['zero_rel_var']))
                mean_rel_vars.append(float(data['mean_rel_var']))
            elif 'zero_snr' in data:
                zero_rel_vars.append(float(data['zero_snr']))
                mean_rel_vars.append(float(data['mean_snr']))

    if not correlations:
        return None

    result = {
        'mean_correlation': float(np.mean(correlations)),
        'std_correlation': float(np.std(correlations)),
        'per_seed_correlation': correlations,
        'n_seeds': len(correlations),
    }
    if zero_rel_vars:
        result['mean_zero_rel_var'] = float(np.mean(zero_rel_vars))
        result['mean_mean_rel_var'] = float(np.mean(mean_rel_vars))
    return result


def compile_exp7_null_control(base_dir, seeds):
    """Aggregate Experiment 7 (Null Control) results."""
    p_values_z = []
    p_values_r = []
    real_sig_z = []
    real_sig_r = []

    for seed in seeds:
        path = os.path.join(base_dir, f'seed_{seed}', 'experiments',
                            'experiment_7_null_control',
                            'experiment_7_null_control.pt')
        data = _safe_load_pt(path)
        if data:
            p_values_z.append(float(data['p_value_z']))
            p_values_r.append(float(data['p_value_r']))
            real_sig_z.append(int(data['real_sig_z']))
            real_sig_r.append(int(data['real_sig_r']))

    if not p_values_z:
        return None

    return {
        'mean_p_value_z': float(np.mean(p_values_z)),
        'mean_p_value_r': float(np.mean(p_values_r)),
        'all_significant_z': all(p < 0.05 for p in p_values_z),
        'all_significant_r': all(p < 0.05 for p in p_values_r),
        'real_sig_z': real_sig_z,
        'real_sig_r': real_sig_r,
        'n_seeds': len(p_values_z),
    }


def compile_exp8_knockout(base_dir, seeds):
    """Aggregate Experiment 8 (Memory Knockout) results."""
    knockout_effects = []
    control_effects = []
    knockout_accs = []
    clean_accs = []
    p_values = []

    for seed in seeds:
        path = os.path.join(base_dir, f'seed_{seed}', 'experiments',
                            'experiment_8_memory_knockout',
                            'experiment_8_memory_knockout.pt')
        data = _safe_load_pt(path)
        if data:
            knockout_effects.append(float(data['mean_knockout_effect']))
            control_effects.append(float(data['mean_control_effect']))
            knockout_accs.append(float(data['knockout_accuracy']))
            clean_accs.append(float(data['clean_accuracy']))
            p_values.append(float(data['p_value_one_sided']))

    if not knockout_effects:
        return None

    return {
        'knockout_effect': {
            'mean': float(np.mean(knockout_effects)),
            'std': float(np.std(knockout_effects)),
            'per_seed': knockout_effects,
        },
        'control_effect': {
            'mean': float(np.mean(control_effects)),
            'std': float(np.std(control_effects)),
            'per_seed': control_effects,
        },
        'knockout_accuracy': {
            'mean': float(np.mean(knockout_accs)),
            'per_seed': knockout_accs,
        },
        'clean_accuracy': {
            'mean': float(np.mean(clean_accs)),
            'per_seed': clean_accs,
        },
        'p_values': p_values,
        'all_significant': all(p < 0.05 for p in p_values),
        'n_seeds': len(knockout_effects),
    }


def compile_audit(base_dir, seeds):
    """Aggregate shortcut audit results."""
    all_passed = []

    for seed in seeds:
        # Check for audit results
        run_dir = os.path.join(base_dir, f'seed_{seed}')
        audit_path = os.path.join(run_dir, 'audit_results.json')
        data = _safe_load_json(audit_path)
        if data and 'passed' in data:
            all_passed.append(data['passed'])

    if not all_passed:
        return None

    return {
        'all_passed': all(all_passed),
        'per_seed': all_passed,
        'pass_rate': sum(all_passed) / len(all_passed),
        'n_seeds': len(all_passed),
    }


def compile_all(base_dir, seeds):
    """Master compilation function — called by run_all_seeds.py."""
    print("\n" + "=" * 70)
    print("  CROSS-SEED RESULT COMPILATION")
    print("=" * 70)
    print(f"  Base dir: {base_dir}")
    print(f"  Seeds:    {seeds}")
    print()

    compiled = {
        'seeds': seeds,
        'n_seeds': len(seeds),
    }

    # ── Training metrics ─────────────────────────────────────────
    print("  Compiling training metrics...")
    tgn = compile_training_metrics(base_dir, seeds)
    if tgn:
        compiled['tgn_training'] = tgn
        _print_metric_block("TGN Training", tgn)

    # ── Flat-GRU metrics ─────────────────────────────────────────
    print("  Compiling Flat-GRU metrics...")
    flat = compile_flatgru_metrics(base_dir, seeds)
    if flat:
        compiled['flatgru_training'] = flat
        _print_metric_block("Flat-GRU Training", flat)

    # ── TGN vs Flat-GRU comparison ───────────────────────────────
    if tgn and flat and 'roc_auc' in tgn and 'roc_auc' in flat:
        gap = tgn['roc_auc']['mean'] - flat['roc_auc']['mean']
        compiled['tgn_vs_flatgru'] = {
            'roc_auc_gap': float(gap),
            'tgn_better': gap > 0,
        }
        print(f"\n  TGN vs Flat-GRU ROC-AUC gap: {gap:+.4f} "
              f"({'TGN wins' if gap > 0 else 'Flat-GRU wins'})")

    # ── Shortcut audit ───────────────────────────────────────────
    print("\n  Compiling audit results...")
    audit = compile_audit(base_dir, seeds)
    if audit:
        compiled['audit'] = audit
        status = "ALL PASSED" if audit['all_passed'] else f"SOME FAILED ({audit['pass_rate']:.0%})"
        print(f"    Audit: {status}")

    # ── Experiment 1 ─────────────────────────────────────────────
    print("\n  Compiling Experiment 1 (Causal Map)...")
    exp1 = compile_exp1_causal_map(base_dir, seeds)
    if exp1:
        compiled['exp1_causal_map'] = exp1
        print(f"    Compiled from {exp1['n_seeds']} seeds")

    # ── Experiment 2 ─────────────────────────────────────────────
    print("  Compiling Experiment 2 (Composition)...")
    exp2 = compile_exp2_composition(base_dir, seeds)
    if exp2:
        compiled['exp2_composition'] = exp2
        r = exp2['recovery_ratio']
        print(f"    Recovery ratio: {r['mean']:.4f} ± {r['std']:.4f}")

    # ── Experiment 2 permutation test ────────────────────────────
    print("  Compiling Experiment 2 (Permutation Test)...")
    perm = compile_exp2_permutation(base_dir, seeds)
    if perm:
        compiled['exp2_permutation'] = perm
        print(f"    Mean p-value: {perm['mean_p_value']:.6f}")
        print(f"    All significant (p<0.05): {perm['all_significant_005']}")
        print(f"    All significant (p<0.001): {perm['all_significant_001']}")

    # ── Experiment 3A ────────────────────────────────────────────
    print("  Compiling Experiment 3A (Gate Analysis)...")
    exp3 = compile_exp3_gate_analysis(base_dir, seeds)
    if exp3:
        compiled['exp3_gate_analysis'] = exp3
        if 'dprime_z' in exp3:
            print(f"    d'(z): {exp3['dprime_z']['mean_overall']:.4f} ± {exp3['dprime_z']['std_overall']:.4f}")
        if 'dprime_r' in exp3:
            print(f"    d'(r): {exp3['dprime_r']['mean_overall']:.4f} ± {exp3['dprime_r']['std_overall']:.4f}")

    # ── Experiment 3B ────────────────────────────────────────────
    print("  Compiling Experiment 3B (Causal Gates)...")
    exp3b = compile_exp3b_causal_gate(base_dir, seeds)
    if exp3b:
        compiled['exp3b_causal_gates'] = exp3b
        for key in ['clamp_z', 'clamp_r', 'clamp_both', 'shuffle_z']:
            if key in exp3b:
                e = exp3b[key]
                label = f"{key} (control)" if key == 'shuffle_z' else key
                print(f"    {label}: {e['mean_effect']:.4f} ± {e['std_effect']:.4f}")

    # ── Experiment 4 ─────────────────────────────────────────────
    print("  Compiling Experiment 4 (Time Window)...")
    exp4 = compile_exp4_time_window(base_dir, seeds)
    if exp4:
        compiled['exp4_time_window'] = exp4
        tp = exp4['transition_point']
        print(f"    Transition Δt: {tp['mean']:.2f} ± {tp['std']:.2f}")

    # ── Experiment 5 ─────────────────────────────────────────────
    print("  Compiling Experiment 5 (Source Shuffle)...")
    exp5 = compile_exp5_source_shuffle(base_dir, seeds)
    if exp5:
        compiled['exp5_source_shuffle'] = exp5
        print(f"    Cosine sim: {exp5['mean_cosine_similarity']:.4f} ± {exp5['std_cosine_similarity']:.4f}")

    # ── Experiment 6 (Ablation Comparison) ─────────────────────────
    print("  Compiling Experiment 6 (Ablation Comparison)...")
    exp6 = compile_exp6_ablation(base_dir, seeds)
    if exp6:
        compiled['exp6_ablation_comparison'] = exp6
        print(f"    Correlation (zero vs mean): {exp6['mean_correlation']:.4f} ± {exp6['std_correlation']:.4f}")

    # ── Experiment 7 (Null Control) ──────────────────────────────
    print("  Compiling Experiment 7 (Null Control)...")
    exp7 = compile_exp7_null_control(base_dir, seeds)
    if exp7:
        compiled['exp7_null_control'] = exp7
        print(f"    z_t null p-value: {exp7['mean_p_value_z']:.4f}")
        print(f"    r_t null p-value: {exp7['mean_p_value_r']:.4f}")

    # ── Experiment 8 (Memory Knockout) ───────────────────────────
    print("  Compiling Experiment 8 (Memory Knockout)...")
    exp8 = compile_exp8_knockout(base_dir, seeds)
    if exp8:
        compiled['exp8_memory_knockout'] = exp8
        print(f"    Knockout effect: {exp8['knockout_effect']['mean']:.4f} ± {exp8['knockout_effect']['std']:.4f}")
        print(f"    Knockout accuracy: {exp8['knockout_accuracy']['mean']:.1%}")

    # ── Experiment 9 (Cross-Seed Consistency) ────────────────────
    if len(seeds) >= 2:
        print("  Running Experiment 9 (Cross-Seed Consistency)...")
        try:
            from experiments.static.exp9_cross_seed_consistency import run as run_exp9
            exp9_results = run_exp9(base_dir, seeds, output_dir=base_dir)
            if exp9_results:
                compiled['exp9_cross_seed'] = {
                    k: v for k, v in exp9_results.items()
                    if not isinstance(v, (torch.Tensor,))
                }
        except Exception as e:
            print(f"    ⚠ Exp9 failed: {e}")

    # ── Save compiled results ────────────────────────────────────
    out_path = os.path.join(base_dir, 'compiled_results.json')
    with open(out_path, 'w') as f:
        json.dump(compiled, f, indent=2, default=str)
    print(f"\n  ✓ Compiled results saved to: {out_path}")

    # ── Print summary table ──────────────────────────────────────
    _print_summary_table(compiled)

    return compiled


def _print_metric_block(title, metrics):
    """Print a formatted block of training metrics."""
    print(f"\n    {title}:")
    for k, v in metrics.items():
        print(f"      {k}: {v['mean']:.4f} ± {v['std']:.4f}  (n={v['n']})")


def _print_summary_table(compiled):
    """Print a final summary table suitable for the paper."""
    print(f"\n\n{'━' * 70}")
    print("  PAPER-READY SUMMARY TABLE")
    print(f"{'━' * 70}")
    print(f"  {'Metric':<40s} {'Value':<25s} {'Seeds':>5s}")
    print(f"  {'─' * 70}")

    if 'tgn_training' in compiled and 'roc_auc' in compiled['tgn_training']:
        t = compiled['tgn_training']['roc_auc']
        print(f"  {'TGN ROC-AUC':<40s} {t['mean']:.4f} ± {t['std']:.4f}       {t['n']:>5d}")

    if 'flatgru_training' in compiled and 'roc_auc' in compiled['flatgru_training']:
        t = compiled['flatgru_training']['roc_auc']
        print(f"  {'Flat-GRU ROC-AUC':<40s} {t['mean']:.4f} ± {t['std']:.4f}       {t['n']:>5d}")

    if 'tgn_vs_flatgru' in compiled:
        gap = compiled['tgn_vs_flatgru']['roc_auc_gap']
        print(f"  {'TGN − Flat-GRU gap':<40s} {gap:+.4f}")

    if 'exp2_composition' in compiled:
        r = compiled['exp2_composition']['recovery_ratio']
        print(f"  {'Composition recovery ratio':<40s} {r['mean']:.4f} ± {r['std']:.4f}       {compiled['exp2_composition']['n_seeds']:>5d}")

    if 'exp2_permutation' in compiled:
        p = compiled['exp2_permutation']
        sig = "***" if p['all_significant_001'] else ("**" if p['all_significant_005'] else "n.s.")
        print(f"  {'Permutation test p-value':<40s} {p['mean_p_value']:.6f} {sig}")

    if 'exp4_time_window' in compiled:
        tp = compiled['exp4_time_window']['transition_point']
        print(f"  {'Time-window transition Δt':<40s} {tp['mean']:.2f} ± {tp['std']:.2f}       {compiled['exp4_time_window']['n_seeds']:>5d}")

    if 'exp5_source_shuffle' in compiled:
        cs = compiled['exp5_source_shuffle']
        print(f"  {'Source-shuffle cosine sim':<40s} {cs['mean_cosine_similarity']:.4f} ± {cs['std_cosine_similarity']:.4f}       {cs['n_seeds']:>5d}")

    if 'exp6_ablation_comparison' in compiled:
        e6 = compiled['exp6_ablation_comparison']
        print(f"  {'Ablation method correlation':<40s} {e6['mean_correlation']:.4f} ± {e6['std_correlation']:.4f}       {e6['n_seeds']:>5d}")

    if 'exp7_null_control' in compiled:
        e7 = compiled['exp7_null_control']
        z_sig = "✓" if e7['all_significant_z'] else "✗"
        r_sig = "✓" if e7['all_significant_r'] else "✗"
        print(f"  {'Null control validated (z_t / r_t)':<40s} {z_sig} / {r_sig}                 {e7['n_seeds']:>5d}")

    if 'exp8_memory_knockout' in compiled:
        e8 = compiled['exp8_memory_knockout']
        ko = e8['knockout_effect']
        print(f"  {'Memory knockout effect':<40s} {ko['mean']:.4f} ± {ko['std']:.4f}       {e8['n_seeds']:>5d}")
        print(f"  {'Knockout accuracy drop':<40s} {e8['clean_accuracy']['mean']:.1%} → {e8['knockout_accuracy']['mean']:.1%}")

    if 'exp9_cross_seed' in compiled and 'map_correlation' in compiled['exp9_cross_seed']:
        mc = compiled['exp9_cross_seed']['map_correlation']
        print(f"  {'Cross-seed map correlation':<40s} {mc['mean']:.4f}")

    if 'audit' in compiled:
        a = compiled['audit']
        print(f"  {'Shortcut audit pass rate':<40s} {a['pass_rate']:.0%}                {a['n_seeds']:>5d}")

    print(f"  {'─' * 70}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Compile cross-seed results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--base_dir', type=str, default='runs',
                        help='Base directory containing seed_* subdirectories')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45],
                        help='Seeds to compile (default: 42 43 44 45)')
    args = parser.parse_args()

    compile_all(args.base_dir, args.seeds)


if __name__ == '__main__':
    main()
