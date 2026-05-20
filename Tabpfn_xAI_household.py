"""
TabPFN — Household-Level Transmission Prediction
Training + Inference + Explainability Analysis  (single model, no bagging)
==========================================================================
Task   : Predict whether within-household COVID-19 transmission occurred.
Label  : household_label  (1 = secondary transmission, 0 = none)
Dataset: Household_Level_Dataset_V2/

Changelog vs ensemble xAI:
  - Bagging removed (no performance benefit; simpler SHAP pipeline)
  - Single TabPFNClassifier trained on a balanced subsample ≤ 90 k rows
  - TRAIN_POS_RATIO: set from TabPFN_Train_Household.py grid search output
  - Threshold derived from balanced val F1 (no longer fixed at 0.30)
  - Prior correction applied for natural-distribution test evaluation
  - No TabPFNEnsemble dependency
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report, confusion_matrix,
                             log_loss, balanced_accuracy_score,
                             cohen_kappa_score, matthews_corrcoef,
                             f1_score)
from sklearn.model_selection import train_test_split
import shap
import os
import json
import matplotlib.pyplot as plt
import time
import torch
import warnings
warnings.filterwarnings('ignore')

from tabpfn import TabPFNClassifier

# CONFIGURATION
TASK_NAME    = 'Household Transmission Prediction'
FOLDS_PATH   = 'Household_Level_Dataset'
OUTPUT_DIR   = 'TabPFN_xAI_Household_2'
LABEL_COL    = 'household_label'

DROP_COLS_BASE = ['household_id', 'IndexDate_household']
DELETED_COLS   = ['secondary_cases_count']   # leaky: directly encodes label

# Training balanced subsampling.
# Set TRAIN_POS_RATIO to the best ratio found by TabPFN_Train_Household.py.
# Default 0.50 — update after running the train script.
TRAIN_POS_RATIO    = 0.60    # ← set from grid search result
TABPFN_MAX_SAMPLES = 45_000  # hard ceiling (TabPFN limit buffer)
PREDICT_BATCH_SIZE = 5_000

# xAI test set uses NATURAL distribution (true prevalence).
XAI_TEST_SIZE = 5_000        # stratified natural-distribution sample

# Threshold fallback (used only if pred_csv checkpoint is loaded).
# Update from train-script output when using checkpoint mode.
THRESHOLD_FALLBACK = 0.6

TABPFN_PARAMS = {
    'device':                    'cuda',
    'n_estimators':              8,
    'ignore_pretraining_limits': True,
    'model_path': './tabpfn_weights/tabpfn-v2.5-classifier-v2.5_default.ckpt',
}

# ANALYSIS CONFIGURATION
# GLOBAL_PI_CONFIG removed: with only 88 features, direct SHAP beeswarm is
# tractable and provides both importance ranking and directionality in a
# single step.  PI is retained only for subgroup analysis.

GLOBAL_BEESWARM_CONFIG = {
    # 88 features → can afford a larger, more representative explain set
    'n_explain':      400,   # balanced explain samples (200 pos + 200 neg)
    'n_background':    50,   # KernelExplainer background size
    'max_evals':      200,   # nsamples per KernelExplainer call
    'batch_size':      50,   # samples per checkpoint batch
    'top_n_display':   30,   # features shown in beeswarm plot
}

LOCAL_SHAP_CONFIG = {
    'n_per_outcome':     4,
    'n_per_feat_value':  2,
    'n_top_features':    5,
    'max_evals':       100,
    'n_background':     30,
    'batch_size':       50,
    'max_waterfall':    15,
}

SUBGROUP_CONFIG = {
    'pi_subset_size':       1_000,
    'pi_n_repeats':             3,
    'top_n_display':           20,
    'pi_top_n_features':       50,
    'beeswarm_top_k':           5,
    'beeswarm_n_quartiles':     4,
    'beeswarm_n_per_cell':      4,
    'beeswarm_n':             100,
    'beeswarm_max_evals':     100,
    'beeswarm_n_bg':           30,
    'local_n_per_label':        3,
    'local_n_per_feat':         2,
    'local_n_top_feats':        3,
    'local_max_evals':        100,
    'local_n_background':      30,
    'min_size':                30,
}

# REMAINING TIME ESTIMATION

def estimate_remaining_time(feature_names, n_subgroups=11,
                             global_bee_batches_done=0,
                             global_bee_total_batches=0):
    """
    Estimate wall-clock time for remaining analysis steps.

    Global PI is no longer part of the pipeline (removed: with 88 features,
    direct SHAP beeswarm is used instead).  PI is retained only inside
    subgroup analysis.
    """
    n_feat            = len(feature_names)
    min_per_bee_batch = 5.0
    scale_1k          = 1000.0 / 3000.0
    min_per_call_3k   = 2129.0 / (n_feat * SUBGROUP_CONFIG['pi_n_repeats'])

    bee_remaining = max(0, global_bee_total_batches - global_bee_batches_done)
    bee_min       = bee_remaining * min_per_bee_batch

    print(f"\n{'='*80}")
    print(f"⏱  Remaining Time Estimation  ·  {TASK_NAME}")
    print(f"   (Global PI removed — direct SHAP beeswarm on {n_feat} features)")
    print(f"{'='*80}")

    total_min = 0.0

    print(f"  Global SHAP beeswarm: {bee_min:.0f} min  "
          f"({global_bee_batches_done}/{global_bee_total_batches} batches, "
          f"{bee_remaining} remaining)")
    total_min += bee_min

    local_samples = (LOCAL_SHAP_CONFIG['n_per_outcome'] * 4 +
                     LOCAL_SHAP_CONFIG['n_per_feat_value'] * 2 *
                     LOCAL_SHAP_CONFIG['n_top_features'] * 2)
    local_min     = ((local_samples + LOCAL_SHAP_CONFIG['batch_size'] - 1) //
                     LOCAL_SHAP_CONFIG['batch_size']) * min_per_bee_batch
    print(f"  Global Local SHAP   : {local_min:.0f} min  (~{local_samples} samples)")
    total_min += local_min

    n_sg_feat = SUBGROUP_CONFIG.get('pi_top_n_features', n_feat)
    sg_pi_min_per_sg   = n_sg_feat * SUBGROUP_CONFIG['pi_n_repeats'] * min_per_call_3k * scale_1k
    sg_bee_min_per_sg  = ((SUBGROUP_CONFIG['beeswarm_n'] + 49) // 50) * min_per_bee_batch
    sg_local_min_per_sg= 2 * min_per_bee_batch
    sg_total = (sg_pi_min_per_sg + sg_bee_min_per_sg + sg_local_min_per_sg) * n_subgroups

    print(f"  Subgroup PI features: {n_sg_feat} (filtered from {n_feat})")
    print(f"  Subgroup total ({n_subgroups}): {sg_total:.0f} min  ({sg_total/60:.1f} h)")
    total_min += sg_total

    print(f"\n  📌 Total remaining  : {total_min:.0f} min  ({total_min/60:.1f} h)")
    print(f"  📌 Recommended node : {total_min/60*1.2:.0f} h  (+20% buffer)")
    print(f"{'='*80}\n")
    return total_min

def _sample_with_ratio(df: pd.DataFrame, target_size: int, pos_ratio: float,
                       random_state: int, tag: str = '') -> pd.DataFrame:
    """Balanced subsample at requested pos_ratio."""
    n_pos = int(target_size * pos_ratio)
    n_neg = target_size - n_pos
    pos   = df[df[LABEL_COL] == 1]
    neg   = df[df[LABEL_COL] == 0]
    if len(pos) < n_pos:
        n_pos = len(pos); n_neg = target_size - n_pos
    if len(neg) < n_neg:
        n_neg = len(neg); n_pos = target_size - n_neg
    sampled = pd.concat([
        pos.sample(n=n_pos, replace=False, random_state=random_state),
        neg.sample(n=n_neg, replace=False, random_state=random_state),
    ]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    if tag:
        print(f"    [{tag}] {len(sampled):,} samples  "
              f"pos={sampled[LABEL_COL].mean():.1%}")
    return sampled

def prior_correction(p_model: np.ndarray, p_train: float,
                     p_true: float) -> np.ndarray:
    """Log-odds shift from training prevalence to natural prevalence."""
    eps   = 1e-7
    logit = np.log(np.clip(p_model, eps, 1-eps) /
                   np.clip(1-p_model, eps, 1-eps))
    shift = (np.log(p_true / (1-p_true)) -
             np.log(p_train / (1-p_train)))
    return 1.0 / (1.0 + np.exp(-(logit + shift)))

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                           n_steps: int = 200) -> float:
    """Return the threshold maximising macro F1."""
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.01, 0.99, n_steps):
        f1 = f1_score(y_true, (y_prob >= thr).astype(int),
                      average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr

def load_and_preprocess(fold: int):
    """
    Load household V2 fold and return:

    Training set
    ------------
    Pool train+val, balanced subsample ≤ TABPFN_MAX_SAMPLES rows at
    TRAIN_POS_RATIO positive rate.

    Test set (for xAI / metrics)
    -----------------------------
    Stratified sample of XAI_TEST_SIZE rows from test_df keeping the NATURAL
    class distribution (true household transmission prevalence).
    """
    drop_cols = DROP_COLS_BASE + DELETED_COLS

    train_df = pd.read_csv(f'{FOLDS_PATH}/train_fold_{fold}.csv', encoding='latin1')
    val_df   = pd.read_csv(f'{FOLDS_PATH}/val_fold_{fold}.csv',   encoding='latin1')
    test_df  = pd.read_csv(f'{FOLDS_PATH}/test_fold_{fold}.csv',  encoding='latin1')

    for df in [train_df, val_df, test_df]:
        df.drop(columns=drop_cols, errors='ignore', inplace=True)
        if LABEL_COL in df.columns and LABEL_COL != 'label':
            df.rename(columns={LABEL_COL: 'label'}, inplace=True)

    feature_names = [c for c in train_df.columns if c != 'label']

    # ── Balanced training subsample ───────────────────────────────────────────
    pool_df = pd.concat([train_df, val_df], ignore_index=True)
    natural_pos_rate = pool_df['label'].mean()
    target_train     = min(TABPFN_MAX_SAMPLES, len(pool_df))
    train_sample     = _sample_with_ratio(
        pool_df.rename(columns={'label': LABEL_COL}),
        target_train, TRAIN_POS_RATIO,
        random_state=42, tag=f'Train (fold {fold})')
    X_train = train_sample[feature_names]
    y_train = train_sample[LABEL_COL].values

    # ── Natural-distribution test set ─────────────────────────────────────────
    test_df_lbl = test_df.rename(columns={'label': LABEL_COL})
    if len(test_df_lbl) > XAI_TEST_SIZE:
        test_sample, _ = train_test_split(
            test_df_lbl, train_size=XAI_TEST_SIZE,
            stratify=test_df_lbl[LABEL_COL], random_state=42)
        test_sample = test_sample.reset_index(drop=True)
    else:
        test_sample = test_df_lbl.copy().reset_index(drop=True)

    X_test = test_sample[feature_names]
    y_test = test_sample[LABEL_COL].values
    natural_test_pos_rate = y_test.mean()

    print(f"  Train : {len(X_train):,} (balanced pos={TRAIN_POS_RATIO:.0%})  "
          f"| {len(feature_names):,} features")
    print(f"  Test  : {len(X_test):,} (natural pos={natural_test_pos_rate:.1%})")
    print(f"  Pool natural pos rate : {natural_pos_rate:.1%}")

    return X_train, y_train, X_test, y_test, feature_names, natural_test_pos_rate

# METRICS

def compute_detailed_metrics(y_true, y_pred, y_prob):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        'accuracy':          float((y_true == y_pred).mean()),
        'macro_auc':         float(roc_auc_score(y_true, y_prob)),
        'macro_f1':          float(report.get('macro avg', {}).get('f1-score', np.nan)),
        'weighted_f1':       float(report.get('weighted avg', {}).get('f1-score', np.nan)),
        'log_loss':          float(log_loss(y_true, np.column_stack([1-y_prob, y_prob]))),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'cohen_kappa':       float(cohen_kappa_score(y_true, y_pred)),
        'mcc':               float(matthews_corrcoef(y_true, y_pred)),
        'class_1_auc':       float(roc_auc_score(y_true, y_prob)),
        'class_1_pr_auc':    float(average_precision_score(y_true, y_prob)),
        'class_1_f1':        float(report.get('1', {}).get('f1-score', np.nan)),
        'class_1_recall':    float(report.get('1', {}).get('recall', np.nan)),
        'class_1_precision': float(report.get('1', {}).get('precision', np.nan)),
        'confusion_matrix':  confusion_matrix(y_true, y_pred).tolist(),
    }

def compute_permutation_importance(model, X, y, feature_names,
                                   n_repeats=5, subset_size=None,
                                   random_state=42):
    rng = np.random.default_rng(random_state)

    if subset_size is not None and subset_size < len(X):
        idx   = rng.choice(len(X), size=subset_size, replace=False)
        X_use = X.iloc[idx].reset_index(drop=True)
        y_use = y[idx]
        print(f"    PI subset: {len(X_use):,} / {len(X):,}")
    else:
        X_use = X.reset_index(drop=True)
        y_use = y

    baseline_prob = model.predict_proba(X_use)[:, 1]
    baseline_auc  = roc_auc_score(y_use, baseline_prob)
    print(f"    Baseline AUC: {baseline_auc:.4f}")

    n_features = len(feature_names)
    all_drops  = np.zeros((n_features, n_repeats))

    t0 = time.time()
    for fi, feat in enumerate(feature_names):
        for r in range(n_repeats):
            X_perm       = X_use.copy()
            X_perm[feat] = rng.permutation(X_perm[feat].values)
            perm_prob    = model.predict_proba(X_perm)[:, 1]
            perm_auc     = roc_auc_score(y_use, perm_prob)
            all_drops[fi, r] = baseline_auc - perm_auc

        elapsed = time.time() - t0
        eta     = elapsed / (fi + 1) * (n_features - fi - 1)
        print(f"\r    [{fi+1:3d}/{n_features}] {feat[:35]:35s} "
              f"mean_drop={all_drops[fi].mean():.4f}  "
              f"ETA {eta/60:.1f} min", end='', flush=True)
    print()

    importance_df = pd.DataFrame({
        'feature':         feature_names,
        'mean_importance': all_drops.mean(axis=1),
        'std_importance':  all_drops.std(axis=1),
    }).sort_values('mean_importance', ascending=False).reset_index(drop=True)
    importance_df['rank'] = np.arange(1, len(importance_df) + 1)
    return importance_df

# SHAP WITH BATCH-LEVEL CHECKPOINTS

def compute_shap_small_with_checkpoint(model, X_explain, X_background,
                                       feature_names, checkpoint_dir,
                                       prefix='shap',
                                       max_evals=100, batch_size=50):
    os.makedirs(checkpoint_dir, exist_ok=True)
    n           = len(X_explain)
    n_batch     = (n + batch_size - 1) // batch_size
    batch_files = [os.path.join(checkpoint_dir, f'{prefix}_batch_{i}.npy')
                   for i in range(n_batch)]

    first_pending = 0
    for i, fp in enumerate(batch_files):
        if os.path.exists(fp):
            first_pending = i + 1
        else:
            break

    if first_pending == n_batch:
        print(f"    ✅ SHAP checkpoints complete ({n_batch} batches), loading")
        return np.vstack([np.load(fp) for fp in batch_files])

    if first_pending > 0:
        print(f"    ♻️  Resuming from batch {first_pending+1}/{n_batch}")

    print(f"    SHAP: {n} samples × {max_evals} evals × {len(X_background)} background")

    predict_fn = lambda x: model.predict_proba(x)[:, 1]
    explainer  = shap.KernelExplainer(predict_fn, X_background, link='identity')

    t0 = time.time()
    for i in range(first_pending, n_batch):
        start   = i * batch_size
        end     = min(start + batch_size, n)
        X_batch = X_explain.iloc[start:end]
        pct     = (i + 1) / n_batch * 100
        print(f"\r      batch [{i+1}/{n_batch}] {pct:.0f}%", end='', flush=True)

        sv = explainer.shap_values(X_batch, nsamples=max_evals, silent=True)
        np.save(batch_files[i], sv)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n    → New batches complete, elapsed {elapsed/60:.1f} min")
    return np.vstack([np.load(fp) for fp in batch_files])

def _quartile_label_sample(X, y, pi_feature_order, feature_names,
                            top_k=10, n_quartiles=4, n_per_cell=5,
                            random_state=42):
    rng      = np.random.default_rng(random_state)
    X_reset  = X.reset_index(drop=True)
    chosen   = set()
    top_feats = [f for f in pi_feature_order if f in X_reset.columns][:top_k]

    for feat in top_feats:
        vals = X_reset[feat].values.astype(float)
        boundaries = np.unique(
            np.nanpercentile(vals, np.linspace(0, 100, n_quartiles + 1)))
        if len(boundaries) < 2:
            boundaries = np.array([vals.min()-1e-9, np.median(vals), vals.max()+1e-9])
        buckets = np.digitize(vals, boundaries[1:-1])
        for b in range(len(boundaries) - 1):
            b_mask = buckets == b
            for lab in np.unique(y):
                lab_mask = y == lab
                cell_idx = np.where(b_mask & lab_mask)[0]
                if len(cell_idx) == 0:
                    continue
                k = min(n_per_cell, len(cell_idx))
                chosen.update(rng.choice(cell_idx, k, replace=False).tolist())

    chosen = sorted(chosen)
    print(f"    Beeswarm stratified pool: {len(chosen)} samples")
    return X_reset.iloc[chosen].reset_index(drop=True), y[chosen]

# GLOBAL ANALYSIS

def global_feature_analysis(model, X_test, y_test, feature_names,
                             beeswarm_cfg=GLOBAL_BEESWARM_CONFIG,
                             save_dir=None):
    """
    Direct global SHAP beeswarm — no Permutation Importance prerequisite.

    With only 88 features, KernelExplainer SHAP is tractable directly on a
    class-balanced explain set.  Features are ordered in the beeswarm by
    descending mean|SHAP|, providing both an importance ranking and
    directional information in a single step.

    Checkpointing: SHAP values are saved batch-by-batch so a restart can
    resume without repeating completed work.

    Returns
    -------
    shap_importance_df : DataFrame with columns [feature, mean_importance, rank]
                         (mean_importance = mean|SHAP|; compatible with
                          subgroup_analysis(global_pi_df=...) API)
    shap_matrix        : ndarray (n_explain, n_features)
    X_explain          : DataFrame used for explanation
    """
    print(f"\n{'='*80}")
    print("GLOBAL FEATURE ANALYSIS  (direct SHAP beeswarm)")
    print(f"{'='*80}")

    if save_dir is None:
        save_dir = os.path.join(OUTPUT_DIR, 'global_importance')
    os.makedirs(save_dir, exist_ok=True)

    top_n         = beeswarm_cfg['top_n_display']
    bee_path      = os.path.join(save_dir, 'beeswarm.png')
    shap_csv_path = os.path.join(save_dir, 'shap_feature_importance.csv')
    shap_val_path = os.path.join(save_dir, 'beeswarm_shap_values.csv')
    ckpt_dir      = os.path.join(save_dir, 'shap_checkpoints')

    # ── Build a class-balanced explain set ───────────────────────────────────
    n_explain = beeswarm_cfg['n_explain']
    rng_exp   = np.random.default_rng(1)

    pos_idx = np.where(y_test == 1)[0]
    neg_idx = np.where(y_test == 0)[0]
    n_pos   = min(n_explain // 2, len(pos_idx))
    n_neg   = min(n_explain - n_pos, len(neg_idx))

    chosen  = np.concatenate([
        rng_exp.choice(pos_idx, n_pos, replace=False),
        rng_exp.choice(neg_idx, n_neg, replace=False),
    ])
    rng_exp.shuffle(chosen)
    X_explain = X_test.reset_index(drop=True).iloc[chosen].reset_index(drop=True)
    y_explain = y_test[chosen]
    print(f"  Explain set: {len(X_explain)} samples  "
          f"pos={y_explain.mean():.1%}  neg={(1-y_explain.mean()):.1%}")

    # ── Background set for KernelExplainer ───────────────────────────────────
    try:
        bg_idx, _, _, _ = train_test_split(
            np.arange(len(X_test)), y_test,
            train_size=beeswarm_cfg['n_background'],
            stratify=y_test, random_state=0)
    except Exception:
        bg_idx = np.random.default_rng(0).choice(
            len(X_test), beeswarm_cfg['n_background'], replace=False)
    X_bg = X_test.reset_index(drop=True).iloc[bg_idx].reset_index(drop=True)
    print(f"  Background  : {len(X_bg)} samples")

    # ── SHAP computation with batch-level checkpointing ───────────────────────
    if os.path.exists(shap_val_path):
        print(f"\n  ✅ SHAP values checkpoint found, loading: {shap_val_path}")
        shap_matrix = pd.read_csv(shap_val_path).values
    else:
        print(f"\n  Computing SHAP values "
              f"({len(X_explain)} explain × {beeswarm_cfg['max_evals']} evals "
              f"× {len(X_bg)} background) …")
        t0 = time.time()
        shap_matrix = compute_shap_small_with_checkpoint(
            model, X_explain, X_bg, feature_names,
            checkpoint_dir=ckpt_dir, prefix='global_bee',
            max_evals=beeswarm_cfg['max_evals'],
            batch_size=beeswarm_cfg['batch_size'])
        print(f"  SHAP done in {(time.time()-t0)/60:.1f} min")
        pd.DataFrame(shap_matrix, columns=feature_names).to_csv(
            shap_val_path, index=False)

    # Save X_explain outside the if/else: runs on first execution AND on re-runs
    # where shap_val_path exists but the feature CSV was not yet saved.
    # X_explain is always defined above (before the checkpoint check).
    _exp_feat_path = os.path.join(save_dir, 'beeswarm_explain_features.csv')
    _exp_lbl_path  = os.path.join(save_dir, 'beeswarm_explain_labels.npy')
    if not os.path.exists(_exp_feat_path):
        X_explain.to_csv(_exp_feat_path, index=False)
        np.save(_exp_lbl_path, y_explain)
        print(f"  Explain features saved → {_exp_feat_path}")

    # ── Build importance DataFrame (mean|SHAP|, descending) ──────────────────
    shap_importance_df = pd.DataFrame({
        'feature':         feature_names,
        'mean_importance': np.abs(shap_matrix).mean(axis=0),   # named for compat
        'std_importance':  np.abs(shap_matrix).std(axis=0),
    }).sort_values('mean_importance', ascending=False).reset_index(drop=True)
    shap_importance_df['rank'] = np.arange(1, len(shap_importance_df) + 1)
    shap_importance_df.to_csv(shap_csv_path, index=False)

    # ── Beeswarm plot (features ordered by mean|SHAP|) ───────────────────────
    if os.path.exists(bee_path):
        print(f"\n  ✅ Beeswarm already exists, skipping plot")
    else:
        shap_order    = shap_importance_df['feature'].tolist()[:top_n]
        feat_idx_map  = {f: i for i, f in enumerate(feature_names)}
        ordered_i     = [feat_idx_map[f] for f in shap_order if f in feat_idx_map]
        ordered_names = [feature_names[i] for i in ordered_i]

        fig, ax = plt.subplots(figsize=(12, max(6, len(ordered_names) * 0.40)))
        plt.sca(ax)
        shap.summary_plot(
            shap_matrix[:, ordered_i],
            X_explain[ordered_names],
            feature_names=ordered_names,
            max_display=top_n,
            plot_type='dot',
            show=False)
        ax = plt.gca()
        ax.set_title(
            f'{TASK_NAME}  ·  Global SHAP Beeswarm\n'
            f'(n={len(X_explain)} balanced explain samples, '
            f'{beeswarm_cfg["max_evals"]} evals/sample)\n'
            f'Features ordered by mean |SHAP value| ↓',
            fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(bee_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Beeswarm saved → {bee_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n  Top 10 features (mean |SHAP|):")
    for _, row in shap_importance_df.head(10).iterrows():
        print(f"    {int(row['rank']):2d}. {row['feature']:50s} | "
              f"{row['mean_importance']:.4f} ± {row['std_importance']:.4f}")

    return shap_importance_df, shap_matrix, X_explain

# LOCAL SHAP

def _collect_local_indices(y_test, y_pred, y_prob, feature_vals,
                            top_feature_names, cfg):
    rng    = np.random.default_rng(42)
    chosen = set()

    def pick(mask, n):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return []
        return rng.choice(idx, min(n, len(idx)), replace=False).tolist()

    n  = cfg['n_per_outcome']
    tp = (y_pred == 1) & (y_test == 1)
    fp = (y_pred == 1) & (y_test == 0)
    tn = (y_pred == 0) & (y_test == 0)
    fn = (y_pred == 0) & (y_test == 1)
    for mask in [tp, fp, tn, fn]:
        chosen.update(pick(mask, n))

    n2      = cfg['n_per_feat_value']
    medians = feature_vals.median()
    for feat in top_feature_names[:cfg['n_top_features']]:
        if feat not in feature_vals.columns:
            continue
        high = feature_vals[feat] >= medians[feat]
        low  = ~high
        for val_mask in [high, low]:
            for lab in [0, 1]:
                chosen.update(pick(val_mask.values & (y_test == lab), n2))

    return sorted(chosen)

def local_shap_analysis(model, X_test, y_test, y_pred, y_prob, feature_names,
                        top_feature_names, cfg=LOCAL_SHAP_CONFIG, save_dir=None):
    print(f"\n{'='*80}")
    print("LOCAL SHAP ANALYSIS")
    print(f"{'='*80}")

    if save_dir is None:
        save_dir = os.path.join(OUTPUT_DIR, 'local_shap')
    os.makedirs(save_dir, exist_ok=True)

    npy_path = os.path.join(save_dir, 'local_shap_values.npy')
    idx_path = os.path.join(save_dir, 'local_shap_indices.npy')
    ckpt_dir = os.path.join(save_dir, 'shap_checkpoints')

    indices = _collect_local_indices(
        y_test, y_pred, y_prob, X_test, top_feature_names, cfg)
    print(f"  Selected {len(indices)} samples for local SHAP")

    X_local = X_test.iloc[indices].reset_index(drop=True)
    y_local = y_test[indices]

    if os.path.exists(npy_path) and os.path.exists(idx_path):
        saved_idx = np.load(idx_path).tolist()
        if saved_idx == indices:
            print(f"  ✅ Local SHAP checkpoint found, loading")
            shap_matrix = np.load(npy_path)
        else:
            print(f"  ⚠️  Index mismatch, recomputing")
            bg_idx = np.random.default_rng(0).choice(
                len(X_test), cfg['n_background'], replace=False)
            X_bg = X_test.iloc[bg_idx].reset_index(drop=True)
            shap_matrix = compute_shap_small_with_checkpoint(
                model, X_local, X_bg, feature_names,
                checkpoint_dir=ckpt_dir, prefix='local',
                max_evals=cfg['max_evals'], batch_size=cfg['batch_size'])
            np.save(npy_path, shap_matrix)
            np.save(idx_path, np.array(indices))
    else:
        bg_idx = np.random.default_rng(0).choice(
            len(X_test), cfg['n_background'], replace=False)
        X_bg = X_test.iloc[bg_idx].reset_index(drop=True)
        t0   = time.time()
        shap_matrix = compute_shap_small_with_checkpoint(
            model, X_local, X_bg, feature_names,
            checkpoint_dir=ckpt_dir, prefix='local',
            max_evals=cfg['max_evals'], batch_size=cfg['batch_size'])
        np.save(npy_path, shap_matrix)
        np.save(idx_path, np.array(indices))
        print(f"  Local SHAP done in {(time.time()-t0)/60:.1f} min")

    # Save feature values — needed to reproduce waterfall plots without re-running
    feat_csv = os.path.join(save_dir, 'local_explain_features.csv')
    lbl_npy  = os.path.join(save_dir, 'local_explain_labels.npy')
    if not os.path.exists(feat_csv):
        X_local.to_csv(feat_csv, index=False)
        np.save(lbl_npy, y_local)

    explanations = []
    for i, orig_idx in enumerate(indices):
        shap_vals = shap_matrix[i]
        feat_shap = sorted(zip(feature_names, shap_vals),
                           key=lambda x: x[1], reverse=True)
        y_pred_i = int(y_pred[orig_idx])
        y_true_i = int(y_test[orig_idx])
        outcome  = {(1, 1): 'TP', (1, 0): 'FP',
                    (0, 0): 'TN', (0, 1): 'FN'}.get((y_pred_i, y_true_i), '?')
        explanations.append({
            'sample_index': int(orig_idx), 'outcome': outcome,
            'prediction':   y_pred_i, 'probability': float(y_prob[orig_idx]),
            'true_label':   y_true_i,
            'top_positive': {f: float(s) for f, s in feat_shap[:5]},
            'top_negative': {f: float(s) for f, s in feat_shap[-5:]},
        })

    with open(os.path.join(save_dir, 'local_explanations.json'), 'w') as fh:
        json.dump(explanations, fh, indent=2, ensure_ascii=False)

    plotted = {}
    for i, exp in enumerate(explanations):
        out   = exp['outcome']
        fpath = os.path.join(save_dir, f'waterfall_{out}_sample{exp["sample_index"]}.png')
        if os.path.exists(fpath):
            plotted[out] = plotted.get(out, 0) + 1
            continue
        fig, _ = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_matrix[i],
                base_values=float(y_prob.mean()),
                data=X_local.iloc[i].values,
                feature_names=feature_names),
            max_display=cfg['max_waterfall'], show=False)
        plt.title(f"{TASK_NAME}  ·  {out} | "
                  f"prob={exp['probability']:.3f} | true={exp['true_label']}",
                  fontsize=11)
        plt.tight_layout()
        plt.savefig(fpath, dpi=300, bbox_inches='tight')
        plt.close()
        plotted[out] = plotted.get(out, 0) + 1

    print(f"  Waterfall plots: {dict(plotted)}")
    print(f"  ✓ Local SHAP complete")
    return explanations

# SUBGROUP ANALYSIS

def _define_subgroups(y_test, y_pred, y_prob):
    return {
        'high_risk':      y_prob >= 0.384,
        'moderate_risk':  (y_prob >= 0.128) & (y_prob < 0.384),
        'low_risk':       y_prob < 0.128,
        # 'predicted_pos':  y_pred == 1,
        # 'predicted_neg':  y_pred == 0,
        # 'true_positive':  (y_pred == 1) & (y_test == 1),
        # 'true_negative':  (y_pred == 0) & (y_test == 0),
        # 'false_positive': (y_pred == 1) & (y_test == 0),
        # 'false_negative': (y_pred == 0) & (y_test == 1),
        # 'correct':        y_pred == y_test,
        # 'incorrect':      y_pred != y_test,
    }

def subgroup_analysis(model, X_test, y_test, y_pred, y_prob, feature_names,
                      cfg=SUBGROUP_CONFIG, save_dir=None, global_pi_df=None):
    print(f"\n{'='*80}")
    print("SUBGROUP ANALYSIS")
    print(f"{'='*80}")

    if save_dir is None:
        save_dir = os.path.join(OUTPUT_DIR, 'subgroup_analysis')
    os.makedirs(save_dir, exist_ok=True)

    top_n_feat = cfg.get('pi_top_n_features', len(feature_names))
    if global_pi_df is not None and top_n_feat < len(feature_names):
        global_top_feats = [f for f in global_pi_df['feature'].tolist()[:top_n_feat]
                            if f in feature_names]
        saving_pct = (1 - len(global_top_feats) / len(feature_names)) * 100
        print(f"\n  ★ Subgroup PI: top-{top_n_feat} → "
              f"{len(global_top_feats)} features  (~{saving_pct:.0f}% time saved)")
    else:
        global_top_feats = feature_names
        print(f"\n  Subgroup PI using all {len(feature_names)} features")

    subgroups = _define_subgroups(y_test, y_pred, y_prob)
    print("\n  Subgroup sizes:")
    for name, mask in subgroups.items():
        print(f"    {name:20s}: {mask.sum():6,}  ({mask.mean()*100:5.1f}%)")

    all_pi_rows  = []
    summary_rows = []

    for sg_name, mask in subgroups.items():
        n_sg = mask.sum()
        if n_sg < cfg['min_size']:
            print(f"\n  Skipping {sg_name}: only {n_sg} samples")
            continue

        print(f"\n{'─'*70}")
        print(f"  Subgroup: {sg_name}  (n={n_sg:,})")
        sg_dir = os.path.join(save_dir, sg_name)
        os.makedirs(sg_dir, exist_ok=True)

        X_sg    = X_test[mask].reset_index(drop=True)
        y_sg    = y_test[mask]
        yp_sg   = y_pred[mask]
        prob_sg = y_prob[mask]

        has_both     = len(np.unique(y_sg)) == 2
        feats_for_pi = [f for f in global_top_feats if f in X_sg.columns]
        print(f"    PI feature count: {len(feats_for_pi)}")

        pi_csv = os.path.join(sg_dir, 'pi_importance.csv')
        if os.path.exists(pi_csv):
            print(f"  [1/3] ✅ PI checkpoint found")
            pi_sg = pd.read_csv(pi_csv)
            top_feat_names_sg = pi_sg['feature'].tolist()[:cfg['local_n_top_feats']]
        elif has_both:
            print(f"  [1/3] Permutation Importance …")
            t0    = time.time()
            pi_sg = compute_permutation_importance(
                model, X_sg, y_sg, feats_for_pi,
                n_repeats=cfg['pi_n_repeats'],
                subset_size=min(cfg['pi_subset_size'], n_sg))
            pi_sg['subgroup'] = sg_name
            pi_sg.to_csv(pi_csv, index=False)
            all_pi_rows.append(pi_sg)
            print(f"    PI done in {(time.time()-t0)/60:.1f} min")

            top_n_sg  = min(cfg['top_n_display'], len(pi_sg))
            top_pi_sg = pi_sg.head(top_n_sg)
            fig, ax = plt.subplots(figsize=(10, max(5, top_n_sg * 0.38)))
            ax.barh(range(top_n_sg), top_pi_sg['mean_importance'],
                    xerr=top_pi_sg['std_importance'],
                    color='coral', ecolor='gray', capsize=3, alpha=0.85)
            ax.set_yticks(range(top_n_sg))
            ax.set_yticklabels(top_pi_sg['feature'], fontsize=8)
            ax.invert_yaxis()
            ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
            ax.set_xlabel('Mean AUC drop', fontsize=11)
            ax.set_title(
                f'{sg_name} — PI  n={n_sg:,}\n'
                f'(top-{len(feats_for_pi)} features from global PI)',
                fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(sg_dir, 'pi_bar.png'), dpi=300, bbox_inches='tight')
            plt.close()
            top_feat_names_sg = pi_sg['feature'].tolist()[:cfg['local_n_top_feats']]
        else:
            print(f"  [1/3] PI skipped — single-class subgroup")
            pi_sg             = None
            top_feat_names_sg = feats_for_pi[:cfg['local_n_top_feats']]

        pi_order_sg  = pi_sg['feature'].tolist() if pi_sg is not None else feats_for_pi[:]
        feat_idx_map = {f: i for i, f in enumerate(feature_names)}

        bee_png      = os.path.join(sg_dir, 'beeswarm.png')
        bee_npy      = os.path.join(sg_dir, 'beeswarm_shap_values.npy')
        bee_feat_csv = os.path.join(sg_dir, f'{sg_name}_bee_features.csv')
        ckpt_dir_sg  = os.path.join(sg_dir, 'shap_checkpoints')

        if os.path.exists(bee_png) and os.path.exists(bee_npy) and os.path.exists(bee_feat_csv):
            print(f"  [2/3] ✅ Beeswarm already exists, skipping")
        else:
            print(f"  [2/3] Beeswarm SHAP …")
            n_bg_sg   = min(cfg['beeswarm_n_bg'], n_sg // 2, 30)
            bg_idx_sg = np.random.default_rng(1).choice(n_sg, n_bg_sg, replace=False)
            X_bg_sg   = X_sg.iloc[bg_idx_sg].reset_index(drop=True)

            X_exp_sg, _ = _quartile_label_sample(
                X_sg, y_sg, pi_feature_order=pi_order_sg,
                feature_names=feature_names,
                top_k=cfg['beeswarm_top_k'],
                n_quartiles=cfg['beeswarm_n_quartiles'],
                n_per_cell=cfg['beeswarm_n_per_cell'], random_state=2)
            if len(X_exp_sg) > cfg['beeswarm_n']:
                cap_idx  = np.random.default_rng(3).choice(
                    len(X_exp_sg), cfg['beeswarm_n'], replace=False)
                X_exp_sg = X_exp_sg.iloc[cap_idx].reset_index(drop=True)

            t0 = time.time()
            shap_sg = compute_shap_small_with_checkpoint(
                model, X_exp_sg, X_bg_sg, feature_names,
                checkpoint_dir=ckpt_dir_sg, prefix=f'{sg_name}_bee',
                max_evals=cfg['beeswarm_max_evals'], batch_size=50)
            np.save(bee_npy, shap_sg)
            # Save subgroup feature values for plot reproduction
            X_exp_sg.to_csv(bee_feat_csv, index=False)
            print(f"    Beeswarm done in {(time.time()-t0)/60:.1f} min")

            top_n_sg2        = min(cfg['top_n_display'], len(pi_order_sg))
            ordered_i        = [feat_idx_map[f] for f in pi_order_sg[:top_n_sg2]
                                 if f in feat_idx_map]
            ordered_names_sg = [feature_names[i] for i in ordered_i]
            fig, ax = plt.subplots(figsize=(12, max(5, top_n_sg2 * 0.45)))
            plt.sca(ax)
            shap.summary_plot(
                shap_sg[:, ordered_i], X_exp_sg[ordered_names_sg],
                feature_names=ordered_names_sg,
                max_display=min(cfg['top_n_display'], len(ordered_names_sg)),
                plot_type='dot', show=False)
            plt.gca().set_title(
                f'{sg_name} — Beeswarm  (n={len(X_exp_sg)}, stratified)',
                fontsize=11, fontweight='bold')
            plt.tight_layout()
            plt.savefig(bee_png, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    Beeswarm saved.")

        print(f"  [3/3] Local SHAP …")
        local_sg_dir  = os.path.join(sg_dir, 'local_shap')
        os.makedirs(local_sg_dir, exist_ok=True)
        local_npy     = os.path.join(local_sg_dir, 'local_shap_values.npy')
        local_idx_npy = os.path.join(local_sg_dir, 'local_shap_indices.npy')
        local_ckpt    = os.path.join(local_sg_dir, 'shap_checkpoints')

        local_idx_sg = _collect_local_indices(
            y_sg, yp_sg, prob_sg, X_sg, top_feat_names_sg,
            cfg={'n_per_outcome':    cfg['local_n_per_label'],
                 'n_per_feat_value': cfg['local_n_per_feat'],
                 'n_top_features':   cfg['local_n_top_feats']})

        if len(local_idx_sg) == 0:
            print(f"    No samples selected — skipping")
            summary_rows.append({'subgroup': sg_name, 'size': n_sg,
                                 'local_shap': 'skipped'})
            continue

        X_local_sg  = X_sg.iloc[local_idx_sg].reset_index(drop=True)
        y_local_sg  = y_sg[local_idx_sg]
        yp_local_sg = yp_sg[local_idx_sg]
        pr_local_sg = prob_sg[local_idx_sg]

        if (os.path.exists(local_npy) and os.path.exists(local_idx_npy) and
                np.load(local_idx_npy).tolist() == local_idx_sg):
            print(f"    ✅ Local SHAP checkpoint found")
            shap_local_sg = np.load(local_npy)
        else:
            bg_idx_l = np.random.default_rng(3).choice(
                n_sg, min(cfg['local_n_background'], n_sg), replace=False)
            X_bg_l = X_sg.iloc[bg_idx_l].reset_index(drop=True)
            t0 = time.time()
            shap_local_sg = compute_shap_small_with_checkpoint(
                model, X_local_sg, X_bg_l, feature_names,
                checkpoint_dir=local_ckpt, prefix=f'{sg_name}_local',
                max_evals=cfg['local_max_evals'], batch_size=50)
            np.save(local_npy, shap_local_sg)
            np.save(local_idx_npy, np.array(local_idx_sg))
            print(f"    Local SHAP done in {(time.time()-t0)/60:.1f} min")

        # Save subgroup local feature values for plot reproduction
        sg_feat_csv = os.path.join(local_sg_dir, 'local_explain_features.csv')
        sg_lbl_npy  = os.path.join(local_sg_dir, 'local_explain_labels.npy')
        if not os.path.exists(sg_feat_csv):
            X_local_sg.to_csv(sg_feat_csv, index=False)
            np.save(sg_lbl_npy, y_local_sg)

        local_exps = []
        for i, sg_i in enumerate(local_idx_sg):
            sv       = shap_local_sg[i]
            fsorted  = sorted(zip(feature_names, sv), key=lambda x: x[1], reverse=True)
            y_pred_i = int(yp_local_sg[i])
            y_true_i = int(y_local_sg[i])
            outcome  = {(1, 1): 'TP', (1, 0): 'FP',
                        (0, 0): 'TN', (0, 1): 'FN'}.get((y_pred_i, y_true_i), '?')
            local_exps.append({
                'subgroup_index': int(sg_i), 'outcome': outcome,
                'prediction':     y_pred_i, 'probability': float(pr_local_sg[i]),
                'true_label':     y_true_i,
                'top_positive':   {f: float(s) for f, s in fsorted[:5]},
                'top_negative':   {f: float(s) for f, s in fsorted[-5:]},
            })
            wf_path = os.path.join(local_sg_dir,
                                   f'waterfall_{outcome}_sg{sg_i}.png')
            if not os.path.exists(wf_path):
                fig, _ = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=sv,
                        base_values=float(prob_sg.mean()),
                        data=X_local_sg.iloc[i].values,
                        feature_names=feature_names),
                    max_display=LOCAL_SHAP_CONFIG['max_waterfall'], show=False)
                plt.title(f'{sg_name} | {outcome}  prob={pr_local_sg[i]:.3f}',
                          fontsize=10)
                plt.tight_layout()
                plt.savefig(wf_path, dpi=300, bbox_inches='tight')
                plt.close()

        with open(os.path.join(local_sg_dir, 'local_explanations.json'), 'w') as fh:
            json.dump(local_exps, fh, indent=2, ensure_ascii=False)
        print(f"    Waterfall plots: {len(local_exps)}")

        top1 = top_feat_names_sg[0] if top_feat_names_sg else 'N/A'
        summary_rows.append({'subgroup':            sg_name,
                              'size':               int(n_sg),
                              'top_feature':        top1,
                              'local_shap_samples': len(local_idx_sg)})

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(save_dir, 'subgroup_summary.csv'), index=False)

    print(f"\n  ✓ Subgroup analysis complete")
    return summary_rows

# MAIN PIPELINE

def run_tabpfn_pipeline(fold=1, run_global=True, run_local=True, run_subgroup=True):
    print(f"\n{'='*80}")
    print(f"TabPFN Pipeline  ·  {TASK_NAME}  (single model, no bagging)")
    print(f"{'='*80}")
    t_pipeline = time.time()

    for sub in ['predictions', 'global_importance', 'local_shap', 'subgroup_analysis']:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print(f"\n{'='*80}\nStep 1: Load data\n{'='*80}")
    (X_train, y_train, X_test, y_test,
     feature_names, natural_test_pos_rate) = load_and_preprocess(fold)

    global_shap_path = os.path.join(OUTPUT_DIR, 'global_importance',
                                     'shap_feature_importance.csv')
    ckpt_dir_bee     = os.path.join(OUTPUT_DIR, 'global_importance', 'shap_checkpoints')
    bee_done_batches = len([f for f in os.listdir(ckpt_dir_bee)
                             if f.startswith('global_bee_batch_')]) \
                       if os.path.exists(ckpt_dir_bee) else 0
    bee_total        = (GLOBAL_BEESWARM_CONFIG['n_explain'] +
                        GLOBAL_BEESWARM_CONFIG['batch_size'] - 1) \
                       // GLOBAL_BEESWARM_CONFIG['batch_size']
    estimate_remaining_time(
        feature_names, n_subgroups=11,
        global_bee_batches_done=bee_done_batches,
        global_bee_total_batches=bee_total)

    # ── 2. Train single model ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"Step 2: Train single TabPFNClassifier  "
          f"({len(X_train):,} samples × {len(feature_names)} features, "
          f"pos={TRAIN_POS_RATIO:.0%})")
    print(f"{'='*80}")

    t0 = time.time()
    model = TabPFNClassifier(**TABPFN_PARAMS)
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    print(f"  ✓ Fitted in {fit_time:.1f}s")

    # ── 3. Predict & derive threshold ─────────────────────────────────────────
    print(f"\n{'='*80}\nStep 3: Predictions + threshold derivation\n{'='*80}")
    pred_csv     = os.path.join(OUTPUT_DIR, 'predictions',
                                f'predictions_fold_{fold}.csv')
    metrics_path = os.path.join(OUTPUT_DIR, 'predictions',
                                f'metrics_fold_{fold}.json')

    if os.path.exists(pred_csv):
        print(f"  ✅ Predictions already exist, loading")
        pred_df   = pd.read_csv(pred_csv)
        y_prob    = pred_df['prob_1'].values
        y_pred    = pred_df['prediction'].values
        THRESHOLD = float(pred_df['threshold'].iloc[0]) \
                    if 'threshold' in pred_df.columns else THRESHOLD_FALLBACK
        y_prob_corr = prior_correction(y_prob, TRAIN_POS_RATIO,
                                       natural_test_pos_rate)
    else:
        t0 = time.time()
        if len(X_test) <= PREDICT_BATCH_SIZE:
            y_proba = model.predict_proba(X_test)
        else:
            batches = []
            for start in range(0, len(X_test), PREDICT_BATCH_SIZE):
                end = min(start + PREDICT_BATCH_SIZE, len(X_test))
                batches.append(model.predict_proba(X_test.iloc[start:end]))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            y_proba = np.vstack(batches)
        y_prob = y_proba[:, 1]

        y_prob_corr = prior_correction(y_prob, TRAIN_POS_RATIO,
                                       natural_test_pos_rate)
        THRESHOLD   = find_optimal_threshold(y_test, y_prob_corr)
        print(f"  Derived threshold (corrected probs, macro-F1 optimal): "
              f"{THRESHOLD:.3f}")

        y_pred  = (y_prob_corr >= THRESHOLD).astype(int)
        pred_df = pd.DataFrame({
            'prob_1': y_prob, 'prob_1_corrected': y_prob_corr,
            'prediction': y_pred, 'true_label': y_test,
            'threshold': THRESHOLD,
        })
        pred_df.to_csv(pred_csv, index=False)
        print(f"  ✓ {(time.time()-t0):.1f}s")

    if not os.path.exists(metrics_path):
        metrics = compute_detailed_metrics(y_test, y_pred, y_prob_corr)
        with open(metrics_path, 'w') as fh:
            json.dump(metrics, fh, indent=2)
    else:
        with open(metrics_path) as fh:
            metrics = json.load(fh)

    print(f"  AUC={metrics['macro_auc']:.4f}  F1={metrics['macro_f1']:.4f}  "
          f"threshold={THRESHOLD:.3f}")
    print(f"  Natural test pos rate: {natural_test_pos_rate:.1%}  "
          f"(prior correction applied)")

    xai_results = {}

    # ── 4. Global ────────────────────────────────────────────────────────────
    if run_global:
        print(f"\n{'='*80}\nStep 4: Global Feature Analysis (direct SHAP beeswarm)\n{'='*80}")
        t0 = time.time()
        shap_df, _, _ = global_feature_analysis(
            model, X_test, y_test, feature_names,
            save_dir=os.path.join(OUTPUT_DIR, 'global_importance'))
        xai_results['global_shap'] = shap_df
        print(f"  ✓ {(time.time()-t0)/60:.1f} min")
    else:
        shap_df = pd.read_csv(global_shap_path) if os.path.exists(global_shap_path) \
                  else pd.DataFrame({'feature': feature_names,
                                     'mean_importance': 0,
                                     'rank': range(1, len(feature_names)+1)})

    top_feat_names = shap_df['feature'].tolist()[:LOCAL_SHAP_CONFIG['n_top_features']]

    # ── 5. Local SHAP ─────────────────────────────────────────────────────────
    if run_local:
        print(f"\n{'='*80}\nStep 5: Local SHAP Analysis\n{'='*80}")
        t0 = time.time()
        local_exp = local_shap_analysis(
            model, X_test, y_test, y_pred, y_prob_corr, feature_names,
            top_feature_names=top_feat_names,
            save_dir=os.path.join(OUTPUT_DIR, 'local_shap'))
        xai_results['local'] = local_exp
        print(f"  ✓ {(time.time()-t0)/60:.1f} min")

    # ── 6. Subgroup ────────────────────────────────────────────────────────────
    if run_subgroup:
        print(f"\n{'='*80}\nStep 6: Subgroup Analysis\n{'='*80}")
        t0 = time.time()
        sg_summary = subgroup_analysis(
            model, X_test, y_test, y_pred, y_prob_corr, feature_names,
            save_dir=os.path.join(OUTPUT_DIR, 'subgroup_analysis'),
            global_pi_df=shap_df)
        xai_results['subgroup'] = sg_summary
        print(f"  ✓ {(time.time()-t0)/60:.1f} min")

    total = time.time() - t_pipeline
    print(f"\n{'='*80}")
    print(f"PIPELINE COMPLETE — {total/3600:.2f} h  ({total/60:.0f} min)")
    print(f"Results → {OUTPUT_DIR}")
    print(f"{'='*80}")
    return pred_df, metrics, xai_results

# ENTRY POINT
if __name__ == '__main__':
    if TABPFN_PARAMS.get('device') == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        TABPFN_PARAMS['device'] = 'cpu'
    elif torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pred_df, metrics, results = run_tabpfn_pipeline(
        fold=1,
        run_global=True,
        run_local=True,
        run_subgroup=True,
    )
