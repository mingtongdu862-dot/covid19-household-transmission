# COVID-19 Household Transmission Prediction

**Explainable Machine Learning for Within-Household SARS-CoV-2 Secondary Transmission Risk Prediction**

Master's Thesis Project  
Lund University, 2025

## Overview

This repository contains the complete codebase for a machine learning system that predicts COVID-19 secondary transmission risk within Swedish households using national registry data. The system operates at two complementary levels:

1. **Household Level**: Predicts whether secondary transmission will occur within a household after an index case is detected
2. **Individual Level**: Identifies which specific household members are most susceptible to infection

The project demonstrates the application of transformer-based models (TabPFN) alongside traditional ML approaches, with comprehensive explainability through SHAP analysis and LLM-based interpretation.

## Key Features

- **Multi-level prediction framework** with household and individual risk assessment
- **Multiple ML models**: TabPFN, Logistic Regression, Random Forest, XGBoost
- **Explainable AI**: SHAP-based feature attribution with automated report generation
- **LLM interpretation**: Structured prompts for translating model outputs into domain-expert-accessible insights
- **Swedish registry data**: Demographic, socioeconomic, housing, and medical prescription features

## Dataset Structure

The system processes Swedish national registry data (Statistics Sweden, SCB) from the 2020 COVID-19 pandemic wave.

### Data Levels

#### Household Level
- **Target**: Binary classification (transmission occurred / did not occur)
- **Features**: Aggregated demographics, socioeconomic indicators, housing characteristics, index case statistics
- **Sample size**: ~100k households with confirmed index cases

#### Individual Level  
- **Target**: Binary classification (individual became secondary case / remained uninfected)
- **Features**: Personal demographics, income, living area, medical history (prescription data), homecare usage
- **Conditional**: Only non-index members of households flagged by the household model

### Feature Categories

1. **Demographics**: Age, sex, immigration background, country of birth
2. **Socioeconomic**: Disposable income (individual and household), number of children
3. **Housing**: Living area per person, dwelling type (apartment/villa/etc.)
4. **Healthcare**: Homecare alarm usage (`TRYGG_1`), total homecare contacts (`TRYGG_total`)
5. **Medical History**: Prescription counts for 6 medication categories (ATC codes) selected via chi-square filtering

## Repository Structure

```
.
├── Dataset Creation
│   ├── Household_level_dataset_creator.py    # Creates household-level datasets
│   └── Individual_level_dataset_creator.py   # Creates individual-level datasets
│
├── Model Training
│   ├── LogisticRegression_Train_Household.py
│   ├── LogisticRegression_Train_Individual.py
│   ├── RandomForest_Train_Household.py
│   ├── RandomForest_Train_Individual.py
│   ├── XGBoost_Train_Household.py
│   ├── XGBoost_Train_Individual.py
│   └── Tabpfn_Train_pipeline.py              # TabPFN training with cross-validation
│
└── Explainability & Inference
    ├── Tabpfn_xAI_household.py               # SHAP analysis for household model
    ├── Tabpfn_xAI_individual.py              # SHAP analysis for individual model
    └── TabPFN_Inference_Pipeline.py          # End-to-end inference with report generation
```

## Models

### TabPFN (Primary Model)
- **Type**: Transformer-based in-context learner
- **Architecture**: Tabular Prior-Fitted Network
- **Advantage**: No hand-crafted feature engineering, direct tabular data processing
- **Training**: 5-fold cross-validation with balanced subsampling

### Baseline Models
- **Logistic Regression**: With class weight balancing and SMOTE variants
- **Random Forest**: 200 estimators, balanced class weights
- **XGBoost**: Gradient boosting with scale_pos_weight adjustment

All models handle severe class imbalance (household transmission ~30%, individual susceptibility ~15%).

## Explainability Pipeline

The system generates comprehensive interpretation reports for each household:

1. **SHAP Analysis**: TreeExplainer for feature attribution
2. **Waterfall Plots**: Visual representation of feature contributions
3. **Structured Reports**: Text-based summaries with top influential features
4. **LLM Interpretation**: Chain-of-thought prompts for domain expert translation

### Report Components
- Household composition and model predictions
- Decision thresholds (empirical ML values, not clinical cutoffs)
- Top 20 features ranked by absolute SHAP value
- Individual susceptibility profiles with feature attributions
- Important scope note: Reports describe model behavior, not clinical ground truth

## Usage

### 1. Dataset Creation

```bash
# Create household-level dataset
python Household_level_dataset_creator.py

# Create individual-level dataset  
python Individual_level_dataset_creator.py
```

**Outputs**: 5-fold stratified cross-validation splits (`train_fold_N.csv`, `val_fold_N.csv`, `test_fold_N.csv`)

### 2. Model Training

```bash
# Train TabPFN (recommended)
python Tabpfn_Train_pipeline.py

# Or train baseline models
python LogisticRegression_Train_Household.py
python RandomForest_Train_Individual.py
# etc.
```

**Outputs**: Model checkpoints, performance metrics (AUC, F1, PR-AUC), classification reports

### 3. Explainability Analysis

```bash
# Generate SHAP explanations
python Tabpfn_xAI_household.py
python Tabpfn_xAI_individual.py
```

**Outputs**: SHAP value matrices, feature importance rankings, waterfall plots

### 4. Inference Pipeline

```bash
# Run end-to-end inference with report generation
python TabPFN_Inference_Pipeline.py
```

**Outputs**: 
- `inference_index.json`: Master index of processed households
- Per-household directories with:
  - `report.txt`: Structured text report
  - `summary.json`: Complete model outputs
  - `household_waterfall.png`: Household-level SHAP plot
  - `individual_{pid}_waterfall.png`: Per-member SHAP plots

## Requirements

```
python>=3.9
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
torch>=2.0.0
tabpfn>=0.1.10
xgboost>=1.7.0
shap>=0.42.0
imbalanced-learn>=0.10.0
matplotlib>=3.6.0
tqdm>=4.64.0
```

Install via:
```bash
pip install -r requirements.txt
```

**Note**: TabPFN requires CUDA-capable GPU for optimal performance. CPU inference is supported but slower.

## Key Configuration

### Dataset Creation
- **Class balance**: Stratified sampling maintains original distribution
- **Missing data**: Features with >20% missing values dropped
- **Feature engineering**: Log transformations for income and living area
- **Validation**: 10% held-out test set + 5-fold CV

### TabPFN Training
- **Max training samples**: 50,000 (balanced subsampling)
- **Positive ratio**: 0.50 (grid search optimal)
- **Ensemble size**: 8 estimators
- **Decision thresholds**: Optimized via F1-score on validation set

### SHAP Analysis
- **Background samples**: 30 households/individuals
- **Max evaluations**: 100 per instance
- **Batch size**: 50 for memory efficiency

## Privacy & Data Protection

⚠️ **Important**: This repository contains **code only**. All data files are excluded for privacy compliance.

- Raw data: Swedish national registries (Statistics Sweden, SCB)
- Access: Restricted, requires ethics approval
- De-identification: All person and household IDs are anonymized
- Ground truth: Not included in inference reports (prospective deployment simulation)

## Interpretation Guidelines

**Critical**: Model outputs describe statistical patterns, not clinical diagnoses:
- Probabilities are model scores, not true infection risks
- Thresholds are empirical ML values without clinical meaning
- SHAP values explain model behavior, not causal mechanisms
- All interpretations must be validated by domain experts

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{du2025covid,
  author = {Du, Mingtong},
  title = {Explainable Machine Learning for Within-Household SARS-CoV-2 Secondary Transmission Risk Prediction: A Multi-Level Analysis Using Swedish Population Registers},
  school = {Lund University},
  year = {2025},
  type = {Master's Thesis}
}
```

## References

### Models
- **TabPFN**: Hollmann et al. (2022). "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second." *arXiv:2207.01848*
- **SHAP**: Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions." *NIPS 2017*

### Imbalanced Learning
- Chawla et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *JAIR*
- He & Garcia (2009). "Learning from Imbalanced Data." *IEEE TKDE*

### Domain Context
- Swedish COVID-19 household transmission patterns (2020 wave)
- Swedish national registry infrastructure (SCB, LISA, Patient Register)

## License

This project is released under the MIT License for the code. **Data access is governed by Swedish data protection regulations and requires separate ethics approval.**

## Contact

For questions about the methodology or code:
- **Author**: Mingtong Du
- **Institution**: Lund University
- **Year**: 2025

---

**Disclaimer**: This is a research prototype. Model outputs are not validated for clinical use and should not inform actual COVID-19 response decisions without comprehensive expert review and validation.
