"""
XGBoost – Household-Level Transmission Prediction
==================================================
Task   : Predict whether within-household COVID-19 transmission occurred.
Label  : household_label  (1 = secondary transmission, 0 = none)
Dataset: Household_Level_Dataset_V2/

Improvements over v1
--------------------
1. Feature-distribution report printed after each fold (mean, std, % nonzero
   for the top-N features ranked by XGBoost importance).
2. Optimal classification threshold is found via grid search on the balanced
   validation set (optimising F1 for class 1 by default).  The threshold is
   then applied uniformly to validation AND test hard-classification metrics.
   Threshold-independent metrics (AUC, PR-AUC, log-loss) are unaffected.
3. Naming and labelling are specific to the household-level task throughout.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report, confusion_matrix,
                             log_loss, balanced_accuracy_score,
                             cohen_kappa_score, matthews_corrcoef,
                             f1_score)
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
import xgboost as xgb
import os
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
TASK_NAME  = 'Household Transmission Prediction'
output_dir = 'XGB_results_Household/'
os.makedirs(output_dir, exist_ok=True)

# Columns to drop before training
deleted_cols = ['secondary_cases_count']

# Imbalance handling strategy
# Options: 'class_weight', 'smote', 'borderline_smote', 'adasyn',
#          'undersample', 'smote_tomek', 'smote_enn', 'none'
IMBALANCE_STRATEGY = 'class_weight'
SAMPLING_RATIO     = 0.5       # target minority ratio for sampling methods

# Threshold grid-search configuration
THRESHOLD_CONFIG = {
    'enabled':   True,
    'grid':      np.arange(0.05, 0.95, 0.01),   # thresholds to search
    'metric':    'f1',     # 'f1' | 'mcc' | 'balanced_accuracy'
    # Metric is optimised on the balanced validation set.
    # The best threshold is then applied to both val and test hard predictions.
}

# Feature distribution report
FEAT_DIST_CONFIG = {
    'top_n': 20,    # show distribution for top-N features by importance
}

xgb_params = {
    'n_estimators':    200,
    'max_depth':       8,
    'learning_rate':   0.05,
    'subsample':       0.8,
    'colsample_bytree':0.8,
    'random_state':    42,
    'n_jobs':          -1,
    'objective':       'binary:logistic',
    'eval_metric':     ['auc', 'logloss'],
    'scale_pos_weight':None,   # set dynamically when strategy='class_weight'
    'tree_method':     'hist',
}

# ==================== HELPER FUNCTIONS ====================
def load_and_preprocess(fold: int):
    base_path = 'Household_Level_Dataset'
    drop_cols = ['household_id', 'IndexDate_household'] + deleted_cols
    label_col = 'household_label'

    train_df = pd.read_csv(f'{base_path}/train_fold_{fold}.csv', encoding='latin1')
    val_df   = pd.read_csv(f'{base_path}/val_fold_{fold}.csv',   encoding='latin1')
    test_df  = pd.read_csv(f'{base_path}/test_fold_{fold}.csv',  encoding='latin1')

    for df in [train_df, val_df, test_df]:
        df.drop(columns=drop_cols, errors='ignore', inplace=True)

    y_train = train_df[label_col]
    X_train = train_df.drop(label_col, axis=1)
    y_val   = val_df[label_col]
    X_val   = val_df.drop(label_col, axis=1)
    y_test  = test_df[label_col]
    X_test  = test_df.drop(label_col, axis=1)

    pos_weight    = ((y_train == 0).sum() / (y_train == 1).sum()
                     if (y_train == 1).sum() > 0 else 1.0)
    feature_names = X_train.columns.tolist()
    return X_train, y_train, X_val, y_val, X_test, y_test, pos_weight, feature_names

def apply_imbalance_handling(X_train, y_train, strategy='smote', sampling_ratio=None):
    """Apply chosen imbalance-handling strategy to the training data."""
    print(f"  Original class distribution: "
          f"{dict(zip(*np.unique(y_train, return_counts=True)))}")

    if strategy in ('none', 'class_weight'):
        print(f"  → {'No resampling' if strategy == 'none' else 'class_weight in model'} applied")
        return X_train, y_train

    ss = 'auto' if sampling_ratio is None else sampling_ratio
    samplers = {
        'smote':            SMOTE(sampling_strategy=ss, random_state=42),
        'borderline_smote': BorderlineSMOTE(sampling_strategy=ss, random_state=42),
        'adasyn':           ADASYN(sampling_strategy=ss, random_state=42),
        'undersample':      RandomUnderSampler(sampling_strategy=ss, random_state=42),
        'smote_tomek':      SMOTETomek(sampling_strategy=ss, random_state=42),
        'smote_enn':        SMOTEENN(sampling_strategy=ss, random_state=42),
    }
    if strategy not in samplers:
        raise ValueError(f"Unknown strategy: {strategy}")

    print(f"  → Applying {strategy.upper()} (ratio={ss})")
    X_res, y_res = samplers[strategy].fit_resample(X_train, y_train)
    print(f"  Resampled distribution: {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res

# ── Threshold optimisation ────────────────────────────────────────────────────
def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                           grid: np.ndarray, metric: str) -> tuple:
    """
    Grid-search the classification threshold that maximises `metric` on
    the provided (balanced validation) set.

    Parameters
    ----------
    y_true  : true binary labels
    y_prob  : predicted probabilities for class 1
    grid    : array of candidate thresholds
    metric  : 'f1' | 'mcc' | 'balanced_accuracy'

    Returns
    -------
    best_threshold : float
    best_score     : float
    scores         : dict {threshold: score}  (for diagnostics)
    """
    score_fn = {
        'f1':               lambda yt, yp: f1_score(yt, yp, zero_division=0),
        'mcc':              matthews_corrcoef,
        'balanced_accuracy':balanced_accuracy_score,
    }[metric]

    scores = {}
    for t in grid:
        y_pred = (y_prob >= t).astype(int)
        scores[float(t)] = float(score_fn(y_true, y_pred))

    best_t = max(scores, key=scores.get)
    return best_t, scores[best_t], scores

def compute_detailed_metrics(y_true, y_prob, threshold: float = 0.5, num_classes: int = 2):
    """
    Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true    : true binary labels
    y_prob    : predicted probabilities for class 1
    threshold : decision threshold (applied to y_prob to obtain hard labels)
    """
    y_pred   = (y_prob >= threshold).astype(int)
    report   = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    logloss  = log_loss(y_true, np.column_stack([1 - y_prob, y_prob]))

    result = {
        'threshold':          float(threshold),
        'macro_auc':          float(roc_auc_score(y_true, y_prob)),
        'macro_pr_auc':       float(average_precision_score(y_true, y_prob)),
        'macro_f1':           float(report['macro avg']['f1-score']),
        'weighted_f1':        float(report['weighted avg']['f1-score']),
        'log_loss':           float(logloss),
        'balanced_accuracy':  float(balanced_accuracy_score(y_true, y_pred)),
        'cohen_kappa':        float(cohen_kappa_score(y_true, y_pred)),
        'mcc':                float(matthews_corrcoef(y_true, y_pred)),
        'confusion_matrix':   confusion_matrix(y_true, y_pred).tolist(),
        'classification_report': report,
    }
    for cls in range(num_classes):
        result[f'class_{cls}_precision'] = float(report.get(str(cls), {}).get('precision', np.nan))
        result[f'class_{cls}_recall']    = float(report.get(str(cls), {}).get('recall',    np.nan))
        result[f'class_{cls}_f1']        = float(report.get(str(cls), {}).get('f1-score',  np.nan))
        result[f'class_{cls}_support']   = int  (report.get(str(cls), {}).get('support',   0))
        if cls == 1:
            result[f'class_{cls}_auc']    = float(roc_auc_score(y_true == cls, y_prob))
            result[f'class_{cls}_pr_auc'] = float(average_precision_score(y_true == cls, y_prob))
        else:
            result[f'class_{cls}_auc']    = np.nan
            result[f'class_{cls}_pr_auc'] = np.nan
    return result

# ── Feature distribution report ──────────────────────────────────────────────
def print_feature_distribution(X_train: np.ndarray,
                                feature_names: list,
                                importances: np.ndarray,
                                top_n: int) -> None:
    """
    Print distribution statistics for the top-N features by XGBoost importance.

    For each feature shows: mean, std, median, min, max, and % nonzero
    (useful for diagnosing sparse count features).
    """
    ranked = np.argsort(importances)[::-1][:top_n]

    print(f"\n  {'─'*80}")
    print(f"  TOP {top_n} FEATURES — DISTRIBUTION IN TRAINING DATA")
    print(f"  {'─'*80}")
    header = (f"  {'Rank':<5}{'Feature':<45}"
              f"{'Mean':>8}{'Std':>8}{'Median':>8}"
              f"{'Min':>8}{'Max':>8}{'% >0':>7}")
    print(header)
    print(f"  {'─'*80}")

    for rank, idx in enumerate(ranked, start=1):
        col  = X_train[:, idx]
        col_nonan = col[~np.isnan(col)]
        mean_v   = np.mean(col_nonan)   if len(col_nonan) else np.nan
        std_v    = np.std(col_nonan)    if len(col_nonan) else np.nan
        med_v    = np.median(col_nonan) if len(col_nonan) else np.nan
        min_v    = np.min(col_nonan)    if len(col_nonan) else np.nan
        max_v    = np.max(col_nonan)    if len(col_nonan) else np.nan
        pct_nz   = (col_nonan > 0).mean() * 100 if len(col_nonan) else np.nan

        name = feature_names[idx]
        name_short = (name[:43] + '…') if len(name) > 44 else name

        print(f"  {rank:<5}{name_short:<45}"
              f"{mean_v:>8.3f}{std_v:>8.3f}{med_v:>8.3f}"
              f"{min_v:>8.3f}{max_v:>8.3f}{pct_nz:>6.1f}%")

    print(f"  {'─'*80}")

# ==================== MAIN TRAINING LOOP ====================
print("\n" + "="*80)
print(f"XGBoost  ·  {TASK_NAME}")
print(f"Imbalance strategy : {IMBALANCE_STRATEGY.upper()}")
print(f"Eval set           : full val / test sets (no subsampling)")
print(f"Threshold search   : {'ON  metric=' + THRESHOLD_CONFIG['metric'] if THRESHOLD_CONFIG['enabled'] else 'OFF (fixed 0.50)'}")
print("="*80)

all_fold_results        = []
feature_importance_list = []

for fold in range(1, 6):
    print(f"\n{'='*60}\nFOLD {fold}\n{'='*60}")

    (X_train, y_train, X_val, y_val,
     X_test, y_test, pos_weight, feature_names) = load_and_preprocess(fold)

    # Impute BEFORE any resampling
    imputer     = SimpleImputer(strategy='mean')
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp   = imputer.transform(X_val)
    X_test_imp  = imputer.transform(X_test)

    # Training imbalance handling
    X_tr_bal, y_tr_bal = apply_imbalance_handling(
        X_train_imp, y_train,
        strategy=IMBALANCE_STRATEGY,
        sampling_ratio=SAMPLING_RATIO if IMBALANCE_STRATEGY not in ('none', 'class_weight') else None,
    )

    # Build model
    current_params = xgb_params.copy()
    if IMBALANCE_STRATEGY == 'class_weight':
        current_params['scale_pos_weight'] = pos_weight
        print(f"  → scale_pos_weight = {pos_weight:.2f}")

    print("  Training XGBoost …")
    model = xgb.XGBClassifier(**current_params)
    model.fit(X_tr_bal, y_tr_bal)

    # Raw predicted probabilities
    val_prob  = model.predict_proba(X_val_imp)[:, 1]
    test_prob = model.predict_proba(X_test_imp)[:, 1]

    # ── Threshold grid-search ─────────────────────────────────────────────
    if THRESHOLD_CONFIG['enabled']:
        best_t, best_score, t_scores = find_optimal_threshold(
            y_val.values, val_prob,
            THRESHOLD_CONFIG['grid'],
            THRESHOLD_CONFIG['metric'],
        )
        print(f"\n  Threshold search (metric={THRESHOLD_CONFIG['metric']}):")
        print(f"    Optimal threshold = {best_t:.2f}  "
              f"({THRESHOLD_CONFIG['metric'].upper()} = {best_score:.4f}  "
              f"vs 0.50 baseline = "
              f"{t_scores.get(0.5, float('nan')):.4f})")
    else:
        best_t = 0.50
        t_scores = {}

    # ── Compute metrics with optimal threshold ────────────────────────────
    val_metrics  = compute_detailed_metrics(y_val.values,  val_prob,  threshold=best_t)
    test_metrics = compute_detailed_metrics(y_test.values, test_prob, threshold=best_t)

    # ── Feature distribution (top-N by importance) ────────────────────────
    importances = model.feature_importances_
    print_feature_distribution(
        X_tr_bal, feature_names, importances,
        top_n=FEAT_DIST_CONFIG['top_n'],
    )

    # ── Print per-fold metrics ────────────────────────────────────────────
    for split_name, metrics in [('VALIDATION', val_metrics), ('TEST', test_metrics)]:
        print(f"\n{'='*60}")
        print(f"FOLD {fold} – {split_name} SET  "
              f"(full set | threshold={best_t:.2f})")
        print(f"{'='*60}")
        print(f"  Macro AUC      : {metrics['macro_auc']:.4f}   (threshold-independent)")
        print(f"  PR-AUC         : {metrics['macro_pr_auc']:.4f}   (threshold-independent)")
        print(f"  Log Loss       : {metrics['log_loss']:.4f}   (threshold-independent)")
        print(f"  ─── below metrics use threshold={best_t:.2f} ───")
        print(f"  Macro F1       : {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1    : {metrics['weighted_f1']:.4f}")
        print(f"  Balanced Acc   : {metrics['balanced_accuracy']:.4f}")
        print(f"  Cohen Kappa    : {metrics['cohen_kappa']:.4f}")
        print(f"  MCC            : {metrics['mcc']:.4f}")
        print(f"\n  Class 1 (Household with secondary transmission):")
        print(f"    AUC           : {metrics['class_1_auc']:.4f}")
        print(f"    Precision     : {metrics['class_1_precision']:.4f}")
        print(f"    Recall        : {metrics['class_1_recall']:.4f}")
        print(f"    F1            : {metrics['class_1_f1']:.4f}")
        print(f"    Support       : {metrics['class_1_support']}")
        cm = np.array(metrics['confusion_matrix'])
        print(f"\n  Confusion matrix (threshold={best_t:.2f}):")
        print(f"    TN={cm[0,0]:>6}  FP={cm[0,1]:>6}")
        print(f"    FN={cm[1,0]:>6}  TP={cm[1,1]:>6}")

    # ── Feature importance bookkeeping ────────────────────────────────────
    fold_importance = pd.DataFrame({
        'feature':    feature_names,
        'importance': importances,
        'fold':       fold,
    })
    feature_importance_list.append(fold_importance)

    # Save model artefacts
    joblib.dump(model, os.path.join(output_dir, f'xgb_model_fold_{fold}.joblib'))
    with open(os.path.join(output_dir, f'xgb_features_fold_{fold}.json'), 'w') as f:
        json.dump(feature_names, f)
    with open(os.path.join(output_dir, f'xgb_threshold_fold_{fold}.json'), 'w') as f:
        json.dump({'optimal_threshold': best_t,
                   'threshold_metric':  THRESHOLD_CONFIG['metric'],
                   'threshold_score':   best_score if THRESHOLD_CONFIG['enabled'] else None}, f)

    all_fold_results.append({
        'fold':               fold,
        'n_features':         X_train.shape[1],
        'n_train_original':   int(len(y_train)),
        'n_train_balanced':   int(len(y_tr_bal)),
        'n_val':              int(len(y_val)),
        'n_test':             int(len(y_test)),
        'imbalance_strategy': IMBALANCE_STRATEGY,
        'optimal_threshold':  best_t,
        'threshold_metric':   THRESHOLD_CONFIG['metric'],
        'val':                val_metrics,
        'test':               test_metrics,
    })

# ==================== AGGREGATE AND SAVE ====================
importance_df   = pd.concat(feature_importance_list)
mean_importance = (importance_df.groupby('feature')['importance']
                   .mean().sort_values(ascending=False).reset_index())
top_n     = 50
top_feats = mean_importance.head(top_n)

summary_rows = []
for r in all_fold_results:
    row = {
        'fold':               r['fold'],
        'imbalance_strategy': r['imbalance_strategy'],
        'optimal_threshold':  r['optimal_threshold'],
        'threshold_metric':   r['threshold_metric'],
        'n_train_original':   r['n_train_original'],
        'n_train_balanced':   r['n_train_balanced'],
        # Validation
        'val_macro_auc':    r['val']['macro_auc'],
        'val_pr_auc':       r['val']['macro_pr_auc'],
        'val_macro_f1':     r['val']['macro_f1'],
        'val_weighted_f1':  r['val']['weighted_f1'],
        'val_log_loss':     r['val']['log_loss'],
        'val_balanced_acc': r['val']['balanced_accuracy'],
        'val_kappa':        r['val']['cohen_kappa'],
        'val_mcc':          r['val']['mcc'],
        # Test
        'test_macro_auc':    r['test']['macro_auc'],
        'test_pr_auc':       r['test']['macro_pr_auc'],
        'test_macro_f1':     r['test']['macro_f1'],
        'test_weighted_f1':  r['test']['weighted_f1'],
        'test_log_loss':     r['test']['log_loss'],
        'test_balanced_acc': r['test']['balanced_accuracy'],
        'test_kappa':        r['test']['cohen_kappa'],
        'test_mcc':          r['test']['mcc'],
    }
    for s, split in [('val', r['val']), ('test', r['test'])]:
        for cls in range(2):
            for m in ('auc', 'f1', 'recall', 'precision'):
                row[f'{s}_class{cls}_{m}'] = split.get(f'class_{cls}_{m}', np.nan)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

summary_path     = os.path.join(output_dir, 'household_xgboost_summary.csv')
importance_path  = os.path.join(output_dir, f'household_xgboost_top{top_n}_features.csv')
full_json_path   = os.path.join(output_dir, 'household_xgboost_full_results.json')

summary_df.to_csv(summary_path, index=False)
top_feats.to_csv(importance_path, index=False)
with open(full_json_path, 'w') as f:
    json.dump({
        'task':               TASK_NAME,
        'imbalance_strategy': IMBALANCE_STRATEGY,
        'threshold_config':   {k: (v.tolist() if hasattr(v, 'tolist') else v)
                               for k, v in THRESHOLD_CONFIG.items()},
        'all_folds':          all_fold_results,
        'top_features':       top_feats.to_dict(orient='records'),
    }, f, indent=2)

# ==================== FINAL SUMMARY ====================
print("\n" + "="*80)
print(f"XGBoost  ·  {TASK_NAME}")
print(f"Strategy: {IMBALANCE_STRATEGY.upper()}  |  "
      f"Threshold: grid-search on val (metric={THRESHOLD_CONFIG['metric']})")
print(f"Eval: full val / test sets")
print("="*80)

avg_t = summary_df['optimal_threshold'].mean()
std_t = summary_df['optimal_threshold'].std()
print(f"\n  Optimal threshold (mean ± std across folds): {avg_t:.3f} ± {std_t:.3f}")

for split_label, prefix in [('VALIDATION', 'val'), ('TEST', 'test')]:
    print(f"\n{'='*60}")
    print(f"{split_label} SET — AVERAGED ACROSS 5 FOLDS:")
    print(f"{'='*60}")
    print(f"  Macro AUC       : {summary_df[f'{prefix}_macro_auc'].mean():.4f} ± {summary_df[f'{prefix}_macro_auc'].std():.4f}  (threshold-independent)")
    print(f"  PR-AUC          : {summary_df[f'{prefix}_pr_auc'].mean():.4f} ± {summary_df[f'{prefix}_pr_auc'].std():.4f}  (threshold-independent)")
    print(f"  Log Loss        : {summary_df[f'{prefix}_log_loss'].mean():.4f} ± {summary_df[f'{prefix}_log_loss'].std():.4f}  (threshold-independent)")
    print(f"  ─── below: evaluated at optimal threshold ─────────────")
    print(f"  Macro F1        : {summary_df[f'{prefix}_macro_f1'].mean():.4f} ± {summary_df[f'{prefix}_macro_f1'].std():.4f}")
    print(f"  Weighted F1     : {summary_df[f'{prefix}_weighted_f1'].mean():.4f} ± {summary_df[f'{prefix}_weighted_f1'].std():.4f}")
    print(f"  Balanced Acc    : {summary_df[f'{prefix}_balanced_acc'].mean():.4f} ± {summary_df[f'{prefix}_balanced_acc'].std():.4f}")
    print(f"  Cohen Kappa     : {summary_df[f'{prefix}_kappa'].mean():.4f} ± {summary_df[f'{prefix}_kappa'].std():.4f}")
    print(f"  MCC             : {summary_df[f'{prefix}_mcc'].mean():.4f} ± {summary_df[f'{prefix}_mcc'].std():.4f}")
    print(f"\n  Class 1 (Household with secondary transmission):")
    print(f"    AUC           : {summary_df[f'{prefix}_class1_auc'].mean():.4f} ± {summary_df[f'{prefix}_class1_auc'].std():.4f}")
    print(f"    F1            : {summary_df[f'{prefix}_class1_f1'].mean():.4f} ± {summary_df[f'{prefix}_class1_f1'].std():.4f}")
    print(f"    Recall        : {summary_df[f'{prefix}_class1_recall'].mean():.4f} ± {summary_df[f'{prefix}_class1_recall'].std():.4f}")
    print(f"    Precision     : {summary_df[f'{prefix}_class1_precision'].mean():.4f} ± {summary_df[f'{prefix}_class1_precision'].std():.4f}")

print(f"\n{'='*80}")
print(f"Files saved to {output_dir}:")
print(f"  • {summary_path}")
print(f"  • {importance_path}")
print(f"  • {full_json_path}")
print(f"  • xgb_model_fold_{{1..5}}.joblib")
print(f"  • xgb_threshold_fold_{{1..5}}.json")
print("="*80)
