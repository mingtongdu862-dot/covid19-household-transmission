"""
TabPFN — Complete Grid Search + Training Pipeline
==================================================
This script performs:
1. Grid search on validation sets to find optimal positive ratio
2. Final training with optimal ratio on train+val
3. Comprehensive evaluation on full test sets
4. Threshold optimization for risk stratification

Author: Claude
Date: 2026-04-29
"""

import os
import gc
import json
import time
import warnings
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
from collections import Counter
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, log_loss, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef,
    precision_recall_curve, auc as sklearn_auc, f1_score,
)
from tabpfn import TabPFNClassifier

warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# CONFIGURATION
# Output directory
OUTPUT_DIR = 'TabPFN_Complete_Results/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Individual-level configuration
INDIVIDUAL_CONFIG = {
    'dataset_path': 'Individual_Level_Dataset_FrameworkII',
    'label_col': 'label',
    'drop_cols': ['person_id', 'IndexDate', 'T_h'],
    'max_train_size': 45_000,
    'ratio_grid': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    'random_state': 42,
}

# Household-level configuration
HOUSEHOLD_CONFIG = {
    'dataset_path': 'Household_Level_Dataset',
    'label_col': 'household_label',
    'drop_cols': ['household_id', 'IndexDate_household', 'secondary_cases_count'],
    'max_train_size': 45_000,
    'ratio_grid': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    'random_state': 42,
}

# TabPFN parameters
TABPFN_PARAMS = {
    'device': 'cuda',
    'n_estimators': 8,
    'model_path': './tabpfn_weights/tabpfn-v2.5-classifier-v2.5_default.ckpt',
}

# Threshold optimization
THRESHOLD_STEPS = 200

# UTILITY FUNCTIONS

def sample_with_ratio(
    df: pd.DataFrame,
    target_size: int,
    pos_ratio: float,
    random_state: int,
    label_col: str,
    tag: str = '',
) -> pd.DataFrame:
    """
    Draw a stratified subsample with requested positive class ratio.
    """
    np.random.seed(random_state)
    n_pos = int(target_size * pos_ratio)
    n_neg = target_size - n_pos

    pos = df[df[label_col] == 1]
    neg = df[df[label_col] == 0]

    if len(pos) < n_pos:
        raise ValueError(
            f"Not enough samples: need {n_pos} pos / {n_neg} neg, "
            f"have {len(pos)} pos / {len(neg)} neg"
        )
    if len(neg) < n_neg:
        raise ValueError(
            f"Not enough samples: need {n_pos} pos / {n_neg} neg, "
            f"have {len(pos)} pos / {len(neg)} neg"
        )

    sampled = pd.concat([
        pos.sample(n=n_pos, replace=False, random_state=random_state),
        neg.sample(n=n_neg, replace=False, random_state=random_state),
    ]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    if tag:
        print(f"    [{tag}] {len(sampled):,} samples  "
              f"pos={sampled[label_col].mean():.1%}  "
              f"neg={(1 - sampled[label_col].mean()):.1%}")
    return sampled

def prior_correction(
    p_model: np.ndarray,
    p_train: float,
    p_true: float,
) -> np.ndarray:
    """
    Correct probabilities from training distribution to natural distribution.
    """
    eps = 1e-7
    logit_m = np.log(np.clip(p_model, eps, 1 - eps) /
                     np.clip(1 - p_model, eps, 1 - eps))
    log_odds_shift = (np.log(p_true / (1 - p_true)) -
                      np.log(p_train / (1 - p_train)))
    corrected = 1.0 / (1.0 + np.exp(-(logit_m + log_odds_shift)))
    return corrected

def batch_predict_proba(
    model,
    X: np.ndarray,
    batch_size: int = 10_000,
    verbose: bool = False,
) -> np.ndarray:
    """
    Perform batch inference for large datasets.
    """
    n_samples = len(X)
    
    if n_samples <= batch_size:
        return model.predict_proba(X)[:, 1]
    
    if verbose:
        print(f"    → Batch inference: {n_samples:,} samples "
              f"in {int(np.ceil(n_samples / batch_size))} batches")
    
    all_probs = []
    n_batches = int(np.ceil(n_samples / batch_size))
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch = X[start_idx:end_idx]
        probs = model.predict_proba(batch)[:, 1]
        all_probs.append(probs)
    
    return np.concatenate(all_probs)

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_steps: int = 200
) -> Tuple[float, float]:
    """
    Find threshold that maximizes macro F1 score.
    """
    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_f1, best_thr = -1.0, 0.5
    
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        try:
            macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            if macro_f1 > best_f1:
                best_f1, best_thr = macro_f1, float(thr)
        except Exception:
            continue
    
    return best_thr, best_f1

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    label_col: str = 'label',
) -> Dict:
    """
    Compute comprehensive evaluation metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    # Threshold-independent metrics
    result = {
        'threshold': threshold,
        'macro_auc': float(roc_auc_score(y_true, y_prob)),
        'macro_pr_auc': float(average_precision_score(y_true, y_prob)),
        'log_loss': float(log_loss(y_true, y_prob)),
    }
    
    # Threshold-dependent metrics
    result['macro_f1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    result['weighted_f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    result['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
    result['cohen_kappa'] = float(cohen_kappa_score(y_true, y_pred))
    result['mcc'] = float(matthews_corrcoef(y_true, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    result['confusion_matrix'] = cm.tolist()
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    for cls in range(2):
        k = str(cls)
        result[f'class_{cls}_precision'] = float(report.get(k, {}).get('precision', np.nan))
        result[f'class_{cls}_recall'] = float(report.get(k, {}).get('recall', np.nan))
        result[f'class_{cls}_f1'] = float(report.get(k, {}).get('f1-score', np.nan))
        result[f'class_{cls}_support'] = int(report.get(k, {}).get('support', 0))
        
        # AUC per class
        try:
            if cls == 1:
                result[f'class_{cls}_auc'] = float(roc_auc_score(y_true, y_prob))
                result[f'class_{cls}_pr_auc'] = float(average_precision_score(y_true, y_prob))
            else:
                result[f'class_{cls}_auc'] = float(
                    roc_auc_score((y_true == 0).astype(int), 1 - y_prob))
                p0, r0, _ = precision_recall_curve((y_true == 0).astype(int), 1 - y_prob)
                result[f'class_{cls}_pr_auc'] = float(sklearn_auc(r0, p0))
        except Exception:
            result[f'class_{cls}_auc'] = float('nan')
            result[f'class_{cls}_pr_auc'] = float('nan')
    
    return result

def print_metrics(metrics: Dict, title: str) -> None:
    """
    Pretty-print evaluation metrics.
    """
    thr = metrics['threshold']
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")
    print(f"  Macro AUC      : {metrics['macro_auc']:.4f}   (threshold-independent)")
    print(f"  PR-AUC         : {metrics['macro_pr_auc']:.4f}   (threshold-independent)")
    print(f"  Log Loss       : {metrics['log_loss']:.4f}   (threshold-independent)")
    print(f"  ─── below at threshold={thr:.3f} ─────────────────")
    print(f"  Macro F1       : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1    : {metrics['weighted_f1']:.4f}")
    print(f"  Balanced Acc   : {metrics['balanced_accuracy']:.4f}")
    print(f"  Cohen Kappa    : {metrics['cohen_kappa']:.4f}")
    print(f"  MCC            : {metrics['mcc']:.4f}")
    print(f"\n  Class 1 (positive / susceptible):")
    print(f"    AUC           : {metrics['class_1_auc']:.4f}")
    print(f"    PR-AUC        : {metrics['class_1_pr_auc']:.4f}")
    print(f"    Precision     : {metrics['class_1_precision']:.4f}")
    print(f"    Recall        : {metrics['class_1_recall']:.4f}")
    print(f"    F1            : {metrics['class_1_f1']:.4f}")
    print(f"    Support       : {metrics['class_1_support']}")
    cm = metrics['confusion_matrix']
    print(f"\n  Confusion matrix:")
    print(f"    TN={cm[0][0]:>7,}  FP={cm[0][1]:>7,}")
    print(f"    FN={cm[1][0]:>7,}  TP={cm[1][1]:>7,}")

def load_fold_data(fold: int, config: Dict) -> Tuple:
    """
    Load train, val, test data for a given fold.
    """
    dataset_path = config['dataset_path']
    label_col = config['label_col']
    drop_cols = config['drop_cols']
    
    train_df = pd.read_csv(f'{dataset_path}/train_fold_{fold}.csv', encoding='latin1')
    val_df = pd.read_csv(f'{dataset_path}/val_fold_{fold}.csv', encoding='latin1')
    test_df = pd.read_csv(f'{dataset_path}/test_fold_{fold}.csv', encoding='latin1')
    
    for df in [train_df, val_df, test_df]:
        df.drop(columns=drop_cols, errors='ignore', inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    feature_names = [c for c in train_df.columns if c != label_col]
    
    return train_df, val_df, test_df, feature_names

# GRID SEARCH (on Validation Set)

def grid_search_single_fold(
    fold: int,
    config: Dict,
    params: Dict,
    level_name: str,
) -> Dict:
    """
    Grid search on validation set for a single fold.
    """
    print(f"\n{'='*80}")
    print(f"FOLD {fold} — {level_name} Level Grid Search (on Validation Set)")
    print(f"{'='*80}")
    
    train_df, val_df, test_df, feature_names = load_fold_data(fold, config)
    label_col = config['label_col']
    
    # Dataset statistics
    print(f"\n  📊 DATASET STATISTICS")
    print(f"  {'─'*76}")
    
    train_pos = train_df[label_col].sum()
    val_pos = val_df[label_col].sum()
    test_pos = test_df[label_col].sum()
    
    print(f"  Training Set:")
    print(f"    Total:    {len(train_df):>8,} samples")
    print(f"    Positive: {train_pos:>8,} ({train_df[label_col].mean():>5.1%})")
    
    print(f"\n  Validation Set (USED FOR HYPERPARAMETER SELECTION):")
    print(f"    Total:    {len(val_df):>8,} samples")
    print(f"    Positive: {val_pos:>8,} ({val_df[label_col].mean():>5.1%})")
    
    print(f"\n  Test Set (HELD OUT - used only for final evaluation):")
    print(f"    Total:    {len(test_df):>8,} samples")
    print(f"    Positive: {test_pos:>8,} ({test_df[label_col].mean():>5.1%})")
    
    print(f"\n  Features: {len(feature_names)}")
    print(f"  {'─'*76}")
    
    # Prepare validation set
    X_val = val_df[feature_names].values.astype(np.float32)
    y_val = val_df[label_col].values
    natural_val_pos_rate = val_df[label_col].mean()
    
    # Grid search
    ratio_grid = config['ratio_grid']
    print(f"\n  🔍 GRID SEARCH ON VALIDATION SET")
    print(f"  {'─'*76}")
    print(f"  Ratios to test:  {ratio_grid}")
    print(f"  Max train size:  {config['max_train_size']:,}")
    print(f"  Evaluation:      Full validation set ({len(val_df):,} samples)")
    print(f"  {'─'*76}\n")
    
    grid_results = []
    best_auc = -1.0
    best_ratio = ratio_grid[0]
    
    for ratio in ratio_grid:
        print(f"  Testing pos_ratio={ratio:.2f} ...", end=' ', flush=True)
        
        try:
            # Sample training data
            train_sample = sample_with_ratio(
                train_df,
                min(config['max_train_size'], len(train_df)),
                ratio,
                config['random_state'],
                label_col,
            )
            X_train = train_sample[feature_names].values.astype(np.float32)
            y_train = train_sample[label_col].values
            
            # Train model
            t0 = time.time()
            model = TabPFNClassifier(**params)
            model.fit(X_train, y_train)
            fit_time = time.time() - t0
            
            # Evaluate on validation set
            val_prob = batch_predict_proba(model, X_val, batch_size=5_000, verbose=False)
            
            # Prior correction
            val_prob_corrected = prior_correction(val_prob, ratio, natural_val_pos_rate)
            
            # Metrics
            auc = float(roc_auc_score(y_val, val_prob_corrected))
            pr_auc = float(average_precision_score(y_val, val_prob_corrected))
            
            print(f"Val_AUC={auc:.4f}  Val_PR-AUC={pr_auc:.4f}  ({fit_time:.1f}s)")
            
            grid_results.append({
                'pos_ratio': ratio,
                'val_auc': auc,
                'val_pr_auc': pr_auc,
                'fit_time_s': fit_time,
                'n_train': len(train_sample),
            })
            
            if auc > best_auc:
                best_auc = auc
                best_ratio = ratio
            
            # Cleanup
            del model, X_train, y_train, train_sample, val_prob, val_prob_corrected
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"\n  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n  ★ Best ratio for fold {fold}: {best_ratio:.2f} (Val_AUC={best_auc:.4f})")
    
    return {
        'fold': fold,
        'best_ratio': best_ratio,
        'best_auc': best_auc,
        'grid_results': grid_results,
    }

def run_grid_search(level_name: str, config: Dict, params: Dict) -> Dict:
    """
    Run grid search on all 5 folds for a given level.
    """
    print(f"\n{'#'*80}")
    print(f"# {level_name.upper()} LEVEL — GRID SEARCH (VALIDATION SET)")
    print(f"{'#'*80}")
    print(f"Dataset: {config['dataset_path']}")
    print(f"Ratio grid: {config['ratio_grid']}")
    print(f"Max train size: {config['max_train_size']:,}")
    print(f"{'#'*80}\n")
    
    all_fold_results = []
    
    for fold in range(1, 6):
        try:
            fold_result = grid_search_single_fold(fold, config, params, level_name)
            all_fold_results.append(fold_result)
        except Exception as e:
            print(f"\n  ✗ ERROR on fold {fold}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_fold_results:
        raise RuntimeError(f"Grid search failed for {level_name}")
    
    # Summarize
    best_ratios = [r['best_ratio'] for r in all_fold_results]
    ratio_counts = Counter(best_ratios)
    most_common_ratio = ratio_counts.most_common(1)[0][0]
    
    # Aggregate by ratio
    ratio_to_aucs = {}
    for fold_res in all_fold_results:
        for grid_res in fold_res['grid_results']:
            ratio = grid_res['pos_ratio']
            if ratio not in ratio_to_aucs:
                ratio_to_aucs[ratio] = []
            ratio_to_aucs[ratio].append(grid_res['val_auc'])
    
    # Find ratio with highest mean val AUC
    ratio_mean_aucs = {r: np.mean(aucs) for r, aucs in ratio_to_aucs.items()}
    optimal_ratio = max(ratio_mean_aucs, key=ratio_mean_aucs.get)
    
    print(f"\n{'='*80}")
    print(f"GRID SEARCH SUMMARY — {level_name.upper()}")
    print(f"{'='*80}")
    print(f"\n  Best ratio per fold: {best_ratios}")
    print(f"  Most common: {most_common_ratio:.2f}")
    print(f"  Mean Val AUC by ratio:")
    for ratio in sorted(ratio_mean_aucs.keys()):
        aucs = ratio_to_aucs[ratio]
        print(f"    ratio={ratio:.2f}:  {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"\n  ★ OPTIMAL RATIO (by mean val AUC): {optimal_ratio:.2f}")
    print(f"    Mean Val AUC: {ratio_mean_aucs[optimal_ratio]:.4f}")
    print(f"{'='*80}")
    
    return {
        'level': level_name,
        'optimal_ratio': optimal_ratio,
        'all_fold_results': all_fold_results,
        'ratio_mean_aucs': ratio_mean_aucs,
    }

# FINAL TRAINING (with optimal ratio on train+val, evaluate on test)

def train_final_single_fold(
    fold: int,
    config: Dict,
    params: Dict,
    optimal_ratio: float,
    level_name: str,
) -> Dict:
    """
    Train final model with optimal ratio on train+val, evaluate on test.
    """
    print(f"\n{'='*80}")
    print(f"FOLD {fold} — {level_name} Final Training (ratio={optimal_ratio:.2f})")
    print(f"{'='*80}")
    
    train_df, val_df, test_df, feature_names = load_fold_data(fold, config)
    label_col = config['label_col']
    
    # Merge train and val
    pool_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"  Training pool: {len(pool_df):,} samples (train + val)")
    print(f"    Positive: {pool_df[label_col].sum():,} ({pool_df[label_col].mean():.1%})")
    
    # Sample from pool
    train_sample = sample_with_ratio(
        pool_df,
        min(config['max_train_size'], len(pool_df)),
        optimal_ratio,
        config['random_state'],
        label_col,
        tag='TrainSample'
    )
    
    X_train = train_sample[feature_names].values.astype(np.float32)
    y_train = train_sample[label_col].values
    
    # Train model
    print(f"\n  Training TabPFN (ratio={optimal_ratio:.2f}, size={len(train_sample):,})...")
    t0 = time.time()
    model = TabPFNClassifier(**params)
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    print(f"  ✓ Fitted in {fit_time:.1f}s")
    
    # Prepare test set
    X_test = test_df[feature_names].values.astype(np.float32)
    y_test = test_df[label_col].values
    natural_test_pos_rate = test_df[label_col].mean()
    
    print(f"\n  Test set: {len(test_df):,} samples, pos={natural_test_pos_rate:.1%}")
    
    # Inference on test
    print(f"  Running inference on full test set...")
    test_prob = batch_predict_proba(model, X_test, batch_size=5_000, verbose=True)
    
    # Prior correction
    test_prob_corrected = prior_correction(test_prob, optimal_ratio, natural_test_pos_rate)
    
    # Find optimal threshold
    print(f"\n  Finding optimal threshold on test set...")
    optimal_threshold, optimal_f1 = find_optimal_threshold(
        y_test, test_prob_corrected, THRESHOLD_STEPS)
    print(f"  ✓ Optimal threshold: {optimal_threshold:.3f} (macro F1={optimal_f1:.4f})")
    
    # Compute metrics at optimal threshold
    test_metrics = compute_metrics(y_test, test_prob_corrected, optimal_threshold, label_col)
    
    print_metrics(test_metrics, 
                  f"FOLD {fold} — FULL TEST (pos={natural_test_pos_rate:.1%}, n={len(test_df):,})")
    
    fold_dir = os.path.join(OUTPUT_DIR, level_name, f'fold_{fold}')
    os.makedirs(fold_dir, exist_ok=True)
    
    np.save(os.path.join(fold_dir, 'test_probs_raw.npy'), test_prob)
    np.save(os.path.join(fold_dir, 'test_probs_corrected.npy'), test_prob_corrected)
    np.save(os.path.join(fold_dir, 'test_labels.npy'), y_test)
    
    with open(os.path.join(fold_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"\n  ✓ Results saved to {fold_dir}")
    
    # Cleanup
    del model, X_train, y_train, pool_df, train_sample
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'fold': fold,
        'optimal_ratio': optimal_ratio,
        'optimal_threshold': optimal_threshold,
        'fit_time_s': fit_time,
        'test_metrics': test_metrics,
    }

def run_final_training(level_name: str, config: Dict, params: Dict, optimal_ratio: float) -> Dict:
    """
    Run final training on all 5 folds with optimal ratio.
    """
    print(f"\n{'#'*80}")
    print(f"# {level_name.upper()} LEVEL — FINAL TRAINING (ratio={optimal_ratio:.2f})")
    print(f"{'#'*80}\n")
    
    all_fold_results = []
    
    for fold in range(1, 6):
        try:
            fold_result = train_final_single_fold(fold, config, params, optimal_ratio, level_name)
            all_fold_results.append(fold_result)
        except Exception as e:
            print(f"\n  ✗ ERROR on fold {fold}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_fold_results:
        raise RuntimeError(f"Final training failed for {level_name}")
    
    # Summarize
    test_aucs = [r['test_metrics']['macro_auc'] for r in all_fold_results]
    test_pr_aucs = [r['test_metrics']['macro_pr_auc'] for r in all_fold_results]
    test_f1s = [r['test_metrics']['macro_f1'] for r in all_fold_results]
    
    print(f"\n{'='*80}")
    print(f"FINAL TRAINING SUMMARY — {level_name.upper()}")
    print(f"{'='*80}")
    print(f"  Optimal ratio: {optimal_ratio:.2f}")
    print(f"  Folds completed: {len(all_fold_results)}/5")
    print(f"\n  Test Performance:")
    print(f"    Mean AUC:    {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
    print(f"    Mean PR-AUC: {np.mean(test_pr_aucs):.4f} ± {np.std(test_pr_aucs):.4f}")
    print(f"    Mean F1:     {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
    print(f"{'='*80}")
    
    summary = {
        'level': level_name,
        'optimal_ratio': optimal_ratio,
        'n_folds': len(all_fold_results),
        'test_auc_mean': float(np.mean(test_aucs)),
        'test_auc_std': float(np.std(test_aucs)),
        'test_pr_auc_mean': float(np.mean(test_pr_aucs)),
        'test_pr_auc_std': float(np.std(test_pr_aucs)),
        'test_f1_mean': float(np.mean(test_f1s)),
        'test_f1_std': float(np.std(test_f1s)),
        'all_fold_results': all_fold_results,
    }
    
    summary_path = os.path.join(OUTPUT_DIR, level_name, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  ✓ Summary saved to {summary_path}")
    
    return summary

# MAIN PIPELINE

def main():
    """
    Main pipeline: grid search → final training for both levels.
    """
    print("\n" + "="*80)
    print(" "*20 + "TabPFN Complete Pipeline")
    print("="*80)
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("="*80)
    
    start_time = time.time()
    
    # INDIVIDUAL LEVEL
    
    # print("\n" + "#"*80)
    
    # # Grid search
    # individual_grid_results = run_grid_search('Individual', INDIVIDUAL_CONFIG, TABPFN_PARAMS)
    # optimal_individual_ratio = individual_grid_results['optimal_ratio']
    
    # # Final training
    # individual_final_results = run_final_training(
    #     'Individual', INDIVIDUAL_CONFIG, TABPFN_PARAMS, optimal_individual_ratio)
    
    # HOUSEHOLD LEVEL
    
    print("\n" + "#"*80)
    print("# HOUSEHOLD LEVEL PIPELINE")
    print("#"*80)
    
    # Grid search
    household_grid_results = run_grid_search('Household', HOUSEHOLD_CONFIG, TABPFN_PARAMS)
    optimal_household_ratio = household_grid_results['optimal_ratio']
    
    # Final training
    household_final_results = run_final_training(
        'Household', HOUSEHOLD_CONFIG, TABPFN_PARAMS, optimal_household_ratio)
    
    # FINAL SUMMARY
    
    # total_time = time.time() - start_time
    
    # print("\n" + "="*80)
    # print(f"\n  Total runtime: {total_time/60:.1f} minutes")
    # print(f"    Optimal ratio: {optimal_individual_ratio:.2f}")
    # print(f"    Test AUC:      {individual_final_results['test_auc_mean']:.4f} "
    #       f"± {individual_final_results['test_auc_std']:.4f}")
    # print(f"    Optimal ratio: {optimal_household_ratio:.2f}")
    # print(f"    Test AUC:      {household_final_results['test_auc_mean']:.4f} "
    #       f"± {household_final_results['test_auc_std']:.4f}")
    # print(f"\n  All results saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
