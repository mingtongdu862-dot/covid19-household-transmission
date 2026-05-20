"""
Logistic Regression with robust imbalanced data handling
Multiple imbalance handling strategies available
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             classification_report, confusion_matrix,
                             log_loss, balanced_accuracy_score, 
                             cohen_kappa_score, matthews_corrcoef)
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
import os
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
output_dir = 'LR_results_Household/'
os.makedirs(output_dir, exist_ok=True)

deleted_cols = ['secondary_cases_count']

# Imbalance handling strategy
# Options: 'class_weight', 'smote', 'borderline_smote', 'adasyn', 
#          'undersample', 'smote_tomek', 'smote_enn', 'none'
IMBALANCE_STRATEGY = 'class_weight'

# Target balance ratio for sampling methods (None = auto balance)
SAMPLING_RATIO = 0.8

lr_params = {
    'max_iter': 1000,
    'random_state': 42,
    'n_jobs': -1,
    'solver': 'lbfgs',
    'class_weight': None,  # Will be set to 'balanced' if using class_weight strategy
    'penalty': 'l2',
    'C': 1.0
}

# ==================== HELPER FUNCTIONS ====================
def load_and_preprocess(fold: int):
    base_path = 'Household_Level_Dataset'
    drop_cols = ['household_id', 'IndexDate_household'] + deleted_cols
    label_col = 'household_label'

    train_df = pd.read_csv(f'{base_path}/train_fold_{fold}.csv', encoding='latin1')
    val_df   = pd.read_csv(f'{base_path}/val_fold_{fold}.csv', encoding='latin1')
    test_df  = pd.read_csv(f'{base_path}/test_fold_{fold}.csv', encoding='latin1')

    for df in [train_df, val_df, test_df]:
        df.drop(columns=drop_cols, errors='ignore', inplace=True)

    y_train = train_df[label_col]
    X_train = train_df.drop(label_col, axis=1)
    y_val   = val_df[label_col]
    X_val   = val_df.drop(label_col, axis=1)
    y_test  = test_df[label_col]
    X_test  = test_df.drop(label_col, axis=1)

    feature_names = X_train.columns.tolist()

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names

def apply_imbalance_handling(X_train, y_train, strategy='smote', sampling_ratio=None):
    """Apply various imbalance handling strategies"""
    print(f"  Original class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    
    if strategy == 'none':
        print("  → No resampling applied")
        return X_train, y_train
    
    elif strategy == 'class_weight':
        print("  → Using class_weight in model (no resampling)")
        return X_train, y_train
    
    if sampling_ratio is None:
        sampling_strategy = 'auto'
    else:
        sampling_strategy = sampling_ratio
    
    if strategy == 'smote':
        sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        print(f"  → Applying SMOTE (ratio={sampling_strategy})")
    elif strategy == 'borderline_smote':
        sampler = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=42)
        print(f"  → Applying Borderline SMOTE (ratio={sampling_strategy})")
    elif strategy == 'adasyn':
        sampler = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
        print(f"  → Applying ADASYN (ratio={sampling_strategy})")
    elif strategy == 'undersample':
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        print(f"  → Applying Random Undersampling (ratio={sampling_strategy})")
    elif strategy == 'smote_tomek':
        sampler = SMOTETomek(sampling_strategy=sampling_strategy, random_state=42)
        print(f"  → Applying SMOTE + Tomek Links (ratio={sampling_strategy})")
    elif strategy == 'smote_enn':
        sampler = SMOTEENN(sampling_strategy=sampling_strategy, random_state=42)
        print(f"  → Applying SMOTE + ENN (ratio={sampling_strategy})")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    print(f"  Resampled class distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
    return X_resampled, y_resampled

def compute_detailed_metrics(y_true, y_pred, y_prob, num_classes=2):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    auc_macro    = roc_auc_score(y_true, y_prob)
    pr_auc_macro = average_precision_score(y_true, y_prob)
    cm           = confusion_matrix(y_true, y_pred)
    logloss      = log_loss(y_true, np.column_stack([1-y_prob, y_prob]))
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    kappa        = cohen_kappa_score(y_true, y_pred)
    mcc          = matthews_corrcoef(y_true, y_pred)

    result = {
        'macro_auc':          float(auc_macro),
        'macro_pr_auc':       float(pr_auc_macro),
        'macro_f1':           float(report['macro avg']['f1-score']),
        'weighted_f1':        float(report['weighted avg']['f1-score']),
        'log_loss':           float(logloss),
        'balanced_accuracy':  float(balanced_acc),
        'cohen_kappa':        float(kappa),
        'mcc':                float(mcc),
        'confusion_matrix':   cm.tolist(),
        'classification_report': report,
    }

    for cls in range(num_classes):
        result[f'class_{cls}_precision'] = float(report.get(str(cls), {}).get('precision', np.nan))
        result[f'class_{cls}_recall']    = float(report.get(str(cls), {}).get('recall',    np.nan))
        result[f'class_{cls}_f1']        = float(report.get(str(cls), {}).get('f1-score',  np.nan))
        result[f'class_{cls}_support']   = int  (report.get(str(cls), {}).get('support',   0))
        result[f'class_{cls}_auc']       = roc_auc_score(y_true == cls, y_prob) if cls == 1 else np.nan
        result[f'class_{cls}_pr_auc']    = average_precision_score(y_true == cls, y_prob) if cls == 1 else np.nan

    return result

# ==================== MAIN TRAINING LOOP ====================
print("\n" + "="*80)
print(f"Logistic Regression Training with Imbalance Handling: {IMBALANCE_STRATEGY.upper()}")
print(f"Evaluation: full val / test sets (no subsampling)")
print("="*80)

all_fold_results      = []
feature_importance_list = []

for fold in range(1, 6):
    print(f"\n{'='*60}\nFOLD {fold}\n{'='*60}")
    
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_and_preprocess(fold)
    
    # Impute missing values BEFORE resampling
    imputer      = SimpleImputer(strategy='mean')
    X_train_imp  = imputer.fit_transform(X_train)
    X_val_imp    = imputer.transform(X_val)
    X_test_imp   = imputer.transform(X_test)
    
    # Apply training imbalance handling
    X_train_balanced, y_train_balanced = apply_imbalance_handling(
        X_train_imp, y_train,
        strategy=IMBALANCE_STRATEGY,
        sampling_ratio=SAMPLING_RATIO if IMBALANCE_STRATEGY not in ['none', 'class_weight'] else None
    )
    
    # Set model parameters
    current_params = lr_params.copy()
    if IMBALANCE_STRATEGY == 'class_weight':
        current_params['class_weight'] = 'balanced'
        print(f"  → Set class_weight = 'balanced'")
    
    # Train model
    print(f"Training Logistic Regression...")
    model = LogisticRegression(**current_params)
    model.fit(X_train_balanced, y_train_balanced)
    
    # ── Evaluate on full val / test sets ───────────────────────────────────
    val_prob  = model.predict_proba(X_val_imp)[:, 1]
    val_pred  = model.predict(X_val_imp)
    val_metrics = compute_detailed_metrics(y_val.values, val_pred, val_prob)
    
    test_prob  = model.predict_proba(X_test_imp)[:, 1]
    test_pred  = model.predict(X_test_imp)
    test_metrics = compute_detailed_metrics(y_test.values, test_pred, test_prob)
    
    print(f"\n{'='*60}")
    print(f"FOLD {fold} - VALIDATION SET RESULTS (full val set):")
    print(f"{'='*60}")
    print(f"Macro AUC:        {val_metrics['macro_auc']:.4f}")
    print(f"Macro F1:         {val_metrics['macro_f1']:.4f}")
    print(f"Weighted F1:      {val_metrics['weighted_f1']:.4f}")
    print(f"Log Loss:         {val_metrics['log_loss']:.4f}")
    print(f"Balanced Acc:     {val_metrics['balanced_accuracy']:.4f}")
    print(f"Cohen Kappa:      {val_metrics['cohen_kappa']:.4f}")
    print(f"MCC:              {val_metrics['mcc']:.4f}")
    print(f"\nClass 1 (Has secondary) Metrics:")
    print(f"  AUC:            {val_metrics['class_1_auc']:.4f}")
    print(f"  Precision:      {val_metrics['class_1_precision']:.4f}")
    print(f"  Recall:         {val_metrics['class_1_recall']:.4f}")
    print(f"  F1:             {val_metrics['class_1_f1']:.4f}")
    
    print(f"\n{'='*60}")
    print(f"FOLD {fold} - TEST SET RESULTS (full test set):")
    print(f"{'='*60}")
    print(f"Macro AUC:        {test_metrics['macro_auc']:.4f}")
    print(f"Macro F1:         {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1:      {test_metrics['weighted_f1']:.4f}")
    print(f"Log Loss:         {test_metrics['log_loss']:.4f}")
    print(f"Balanced Acc:     {test_metrics['balanced_accuracy']:.4f}")
    print(f"Cohen Kappa:      {test_metrics['cohen_kappa']:.4f}")
    print(f"MCC:              {test_metrics['mcc']:.4f}")
    print(f"\nClass 1 (Has secondary) Metrics:")
    print(f"  AUC:            {test_metrics['class_1_auc']:.4f}")
    print(f"  Precision:      {test_metrics['class_1_precision']:.4f}")
    print(f"  Recall:         {test_metrics['class_1_recall']:.4f}")
    print(f"  F1:             {test_metrics['class_1_f1']:.4f}")
    
    # Feature importance (coefficients)
    coefficients   = np.abs(model.coef_[0])
    fold_importance = pd.DataFrame({
        'feature':    feature_names,
        'importance': coefficients,
        'fold':       fold,
    })
    feature_importance_list.append(fold_importance)
    
    # Save model and feature names
    model_path    = os.path.join(output_dir, f'lr_model_fold_{fold}.joblib')
    joblib.dump(model, model_path)
    features_path = os.path.join(output_dir, f'lr_features_fold_{fold}.json')
    with open(features_path, 'w') as f:
        json.dump(feature_names, f)
    
    # Store results
    fold_result = {
        'fold':               fold,
        'n_features':         X_train.shape[1],
        'n_train_original':   len(y_train),
        'n_train_balanced':   len(y_train_balanced),
        'n_val':              len(y_val),
        'n_test':             len(y_test),
        'imbalance_strategy': IMBALANCE_STRATEGY,
        'val':                val_metrics,
        'test':               test_metrics,
    }
    all_fold_results.append(fold_result)

# ==================== SAVE RESULTS ====================
importance_df   = pd.concat(feature_importance_list)
mean_importance = importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()
top_n           = 50
top_importance  = mean_importance.head(top_n)

summary_rows = []
for r in all_fold_results:
    row = {
        'fold':               r['fold'],
        'imbalance_strategy': r['imbalance_strategy'],
        'n_train_original':   r['n_train_original'],
        'n_train_balanced':   r['n_train_balanced'],
        # Validation metrics
        'val_macro_auc':    r['val']['macro_auc'],
        'val_macro_f1':     r['val']['macro_f1'],
        'val_weighted_f1':  r['val']['weighted_f1'],
        'val_log_loss':     r['val']['log_loss'],
        'val_balanced_acc': r['val']['balanced_accuracy'],
        'val_kappa':        r['val']['cohen_kappa'],
        'val_mcc':          r['val']['mcc'],
        # Test metrics
        'test_macro_auc':    r['test']['macro_auc'],
        'test_macro_f1':     r['test']['macro_f1'],
        'test_weighted_f1':  r['test']['weighted_f1'],
        'test_log_loss':     r['test']['log_loss'],
        'test_balanced_acc': r['test']['balanced_accuracy'],
        'test_kappa':        r['test']['cohen_kappa'],
        'test_mcc':          r['test']['mcc'],
    }
    for cls in range(2):
        row[f'val_class{cls}_auc']       = r['val'].get(f'class_{cls}_auc',       np.nan)
        row[f'val_class{cls}_f1']        = r['val'].get(f'class_{cls}_f1',        np.nan)
        row[f'val_class{cls}_recall']    = r['val'].get(f'class_{cls}_recall',    np.nan)
        row[f'val_class{cls}_precision'] = r['val'].get(f'class_{cls}_precision', np.nan)
        row[f'test_class{cls}_auc']      = r['test'].get(f'class_{cls}_auc',      np.nan)
        row[f'test_class{cls}_f1']       = r['test'].get(f'class_{cls}_f1',       np.nan)
        row[f'test_class{cls}_recall']   = r['test'].get(f'class_{cls}_recall',   np.nan)
        row[f'test_class{cls}_precision']= r['test'].get(f'class_{cls}_precision',np.nan)
    summary_rows.append(row)

summary_df   = pd.DataFrame(summary_rows)
summary_path = os.path.join(output_dir, 'household_lr_summary.csv')
summary_df.to_csv(summary_path, index=False)

importance_path = os.path.join(output_dir, f'household_lr_top{top_n}_features.csv')
top_importance.to_csv(importance_path, index=False)

full_json_path = os.path.join(output_dir, 'household_lr_full_results.json')
with open(full_json_path, 'w') as f:
    json.dump({
        'imbalance_strategy': IMBALANCE_STRATEGY,
        'all_folds':          all_fold_results,
        'top_features':       top_importance.to_dict(orient='records'),
    }, f, indent=2)

# ==================== PRINT SUMMARY ====================
print("\n" + "="*80)
print(f"Logistic Regression (IMBALANCE: {IMBALANCE_STRATEGY.upper()}) — 5-FOLD CV RESULTS")
print(f"(Evaluated on full val / test sets)")
print("="*80)

print(f"\n{'='*60}")
print("VALIDATION SET — AVERAGED ACROSS 5 FOLDS:")
print(f"{'='*60}")
print(f"Macro AUC:           {summary_df['val_macro_auc'].mean():.4f} ± {summary_df['val_macro_auc'].std():.4f}")
print(f"Macro F1:            {summary_df['val_macro_f1'].mean():.4f} ± {summary_df['val_macro_f1'].std():.4f}")
print(f"Weighted F1:         {summary_df['val_weighted_f1'].mean():.4f} ± {summary_df['val_weighted_f1'].std():.4f}")
print(f"Log Loss:            {summary_df['val_log_loss'].mean():.4f} ± {summary_df['val_log_loss'].std():.4f}")
print(f"Balanced Accuracy:   {summary_df['val_balanced_acc'].mean():.4f} ± {summary_df['val_balanced_acc'].std():.4f}")
print(f"Cohen Kappa:         {summary_df['val_kappa'].mean():.4f} ± {summary_df['val_kappa'].std():.4f}")
print(f"MCC:                 {summary_df['val_mcc'].mean():.4f} ± {summary_df['val_mcc'].std():.4f}")
print(f"\nClass 1 (Has secondary) Metrics:")
print(f"  AUC:               {summary_df['val_class1_auc'].mean():.4f} ± {summary_df['val_class1_auc'].std():.4f}")
print(f"  F1:                {summary_df['val_class1_f1'].mean():.4f} ± {summary_df['val_class1_f1'].std():.4f}")
print(f"  Recall:            {summary_df['val_class1_recall'].mean():.4f} ± {summary_df['val_class1_recall'].std():.4f}")
print(f"  Precision:         {summary_df['val_class1_precision'].mean():.4f} ± {summary_df['val_class1_precision'].std():.4f}")

print(f"\n{'='*60}")
print("TEST SET — AVERAGED ACROSS 5 FOLDS:")
print(f"{'='*60}")
print(f"Macro AUC:           {summary_df['test_macro_auc'].mean():.4f} ± {summary_df['test_macro_auc'].std():.4f}")
print(f"Macro F1:            {summary_df['test_macro_f1'].mean():.4f} ± {summary_df['test_macro_f1'].std():.4f}")
print(f"Weighted F1:         {summary_df['test_weighted_f1'].mean():.4f} ± {summary_df['test_weighted_f1'].std():.4f}")
print(f"Log Loss:            {summary_df['test_log_loss'].mean():.4f} ± {summary_df['test_log_loss'].std():.4f}")
print(f"Balanced Accuracy:   {summary_df['test_balanced_acc'].mean():.4f} ± {summary_df['test_balanced_acc'].std():.4f}")
print(f"Cohen Kappa:         {summary_df['test_kappa'].mean():.4f} ± {summary_df['test_kappa'].std():.4f}")
print(f"MCC:                 {summary_df['test_mcc'].mean():.4f} ± {summary_df['test_mcc'].std():.4f}")
print(f"\nClass 1 (Has secondary) Metrics:")
print(f"  AUC:               {summary_df['test_class1_auc'].mean():.4f} ± {summary_df['test_class1_auc'].std():.4f}")
print(f"  F1:                {summary_df['test_class1_f1'].mean():.4f} ± {summary_df['test_class1_f1'].std():.4f}")
print(f"  Recall:            {summary_df['test_class1_recall'].mean():.4f} ± {summary_df['test_class1_recall'].std():.4f}")
print(f"  Precision:         {summary_df['test_class1_precision'].mean():.4f} ± {summary_df['test_class1_precision'].std():.4f}")

print(f"\n{'='*80}")
print(f"Results saved in {output_dir}:")
print(f"  • Summary CSV          : {summary_path}")
print(f"  • Top {top_n} features CSV : {importance_path}")
print(f"  • Full JSON (all details): {full_json_path}")
print("="*80)
