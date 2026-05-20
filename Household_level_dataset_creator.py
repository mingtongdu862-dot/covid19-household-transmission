"""
Household-Level Dataset Creator for COVID-19 Transmission Prediction
=====================================================================

Creates a binary classification dataset at the household level to predict
whether within-household COVID-19 transmission occurred.

Label Definition
----------------
household_label = 1  →  at least one secondary case (original label=2) exists
household_label = 0  →  no secondary transmission detected

Feature Groups
--------------
1. General household aggregated features (demographic, socioeconomic,
   housing, homecare) — same logic as the original Feature_Household_Engineering.py
   but WITHOUT any contact_/lmed_/ov_/sv_ count features.

2. Index-case features [NEW]:
   Statistical summaries computed exclusively from household members whose
   original person-level label = 1 (co-primary / index cases).
   These capture *who introduced the virus* — a key predictor of onward
   transmission probability.

   Features added:
     index_case_count       — number of index cases in the household
     index_mean_age_2020    — mean age of index cases
     index_min_age_2020     — youngest index case
     index_max_age_2020     — oldest index case
     index_has_elderly      — 1 if any index case ≥ 65 years
     index_has_child        — 1 if any index case < 18 years
     index_proportion_female — proportion of female index cases
     index_proportion_foreign — proportion with foreign background (UtlSvBakg=11)
     index_mean_DispInk04   — mean personal income of index cases
     index_mean_Boarea      — mean living area of index cases
     index_any_TRYGG        — 1 if any index case used homecare services
     index_mean_TRYGG_total — mean homecare service usage among index cases

Input
-----
  Feature_Tables/Raw_Feature_Secondary_Case.csv   (person-level feature table)
  Feature_Tables/household_member.csv             (household membership map)

Output
------
  Household_Level_Dataset_V2/
    train_fold_{1..N}.csv
    val_fold_{1..N}.csv
    test_fold_{1..N}.csv
    folds_info.txt

Author : [Your Name]
Date   : 2025-01-30
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings('ignore')

# CONFIGURATION
class Config:
    """Configuration for household-level dataset creation."""

    # I/O
    PERSON_TABLE_PATH  = 'Feature_Tables/Raw_Feature_Secondary_Case.csv'
    HOUSEHOLD_MAP_PATH = 'Feature_Tables/household_member.csv'
    OUTPUT_DIR         = 'Household_Level_Dataset_V2'
    ENCODING           = 'latin1'

    # Reading
    PERSON_CHUNK_SIZE  = 1_000_000   # rows per chunk when loading person table
    AGGREGATE_BATCH    = 100_000     # households per aggregation batch

    # Imputation
    MISSING_RATE_DROP  = 20          # drop household columns with > 20 % missing

    # K-fold
    TEST_SIZE    = 0.10
    N_FOLDS      = 5
    RANDOM_STATE = 42

    # Testing
    TEST_MODE        = False
    TEST_HOUSEHOLDS  = 500

# Columns to load from the person-level CSV.
# Explicitly excludes contact_/lmed_/ov_/sv_ count features.
PERSON_BASE_COLS = [
    'person_id', 'IndexDate', 'label',
    'UtlSvBakg', 'Fodelseland', 'FodelseArMan', 'Kon',
    'AntalBarnUnder18', 'Boarea_Person', 'Boendeform',
    'DispInk04', 'DispInkFam04',
    'TRYGG_1', 'TRYGG_total',
]

def load_person_table() -> pd.DataFrame:
    """
    Load person-level feature table (base columns only, no count features).

    Returns an indexed DataFrame with person_id as index.
    """
    print(f"\n{'='*70}")
    print("STEP 1 – LOADING PERSON-LEVEL TABLE (base columns only)")
    print(f"{'='*70}")

    # Verify which columns actually exist in the CSV
    header = pd.read_csv(
        Config.PERSON_TABLE_PATH, nrows=0,
        encoding=Config.ENCODING, low_memory=False
    )
    available = set(header.columns)
    use_cols  = [c for c in PERSON_BASE_COLS if c in available]
    missing   = [c for c in PERSON_BASE_COLS if c not in available]
    if missing:
        print(f"  ⚠ Columns not found in CSV (skipped): {missing}")

    del header
    gc.collect()

    chunks = []
    for i, chunk in enumerate(
        pd.read_csv(
            Config.PERSON_TABLE_PATH,
            usecols=use_cols,
            chunksize=Config.PERSON_CHUNK_SIZE,
            encoding=Config.ENCODING,
            low_memory=False,
        )
    ):
        chunk['person_id'] = chunk['person_id'].astype(str).str.rstrip('.0')
        chunk.set_index('person_id', inplace=True)

        # Type conversions
        chunk['IndexDate'] = pd.to_datetime(chunk['IndexDate'], errors='coerce')
        chunk['label']     = pd.to_numeric(chunk['label'], errors='coerce')

        chunks.append(chunk)
        print(f"  Chunk {i+1}: {len(chunk):,} persons loaded")

    df = pd.concat(chunks, ignore_index=False)
    del chunks
    gc.collect()

    # Drop duplicates (keep first)
    df = df[~df.index.duplicated(keep='first')]

    label_dist = df['label'].value_counts().sort_index().to_dict()
    print(f"\n  Total persons  : {len(df):,}")
    print(f"  Label dist.    : {label_dist}")
    print(f"  Columns loaded : {list(df.columns)}")

    return df

# STEP 2 – AGGREGATE TO HOUSEHOLD LEVEL

def compute_household_features(
    household_id: str,
    members: List[str],
    person_df: pd.DataFrame,
) -> Optional[Dict]:
    """
    Compute all household-level features for a single household.

    Validity rules
    --------------
    - Must have ≥ 1 index case (label == 1)
    - Skip if household consists solely of index cases with no other members
      (i.e. no one to potentially transmit to).

    Returns None for invalid households.
    """
    # Identify valid members present in the person table
    valid_members = [m for m in members if m in person_df.index]
    if not valid_members:
        return None

    sub = person_df.loc[valid_members]

    # ── Validity checks ────────────────────────────────────────────────────
    index_cases_cnt     = int((sub['label'] == 1).sum())
    secondary_cases_cnt = int((sub['label'] == 2).sum())
    household_size      = len(sub)

    if index_cases_cnt == 0:
        return None
    # Skip pure-index-case households (no susceptible members present)
    if secondary_cases_cnt == 0 and household_size == index_cases_cnt:
        return None

    # ── Household-level label ──────────────────────────────────────────────
    household_label = 1 if secondary_cases_cnt > 0 else 0

    # ── Age ────────────────────────────────────────────────────────────────
    birth_raw  = pd.to_numeric(sub['FodelseArMan'], errors='coerce')
    birth_year = (birth_raw // 100).astype('Int64')
    age        = (2020 - birth_year).astype('float64')

    # ── Gender ────────────────────────────────────────────────────────────
    kon = pd.to_numeric(sub['Kon'], errors='coerce')
    male_count   = int((kon == 1).sum())
    female_count = int((kon == 2).sum())

    # ── Income ────────────────────────────────────────────────────────────
    disp_ink    = pd.to_numeric(sub['DispInk04'],    errors='coerce')
    disp_ink_fam= pd.to_numeric(sub['DispInkFam04'], errors='coerce')

    # ── TRYGG ─────────────────────────────────────────────────────────────
    trygg_1     = pd.to_numeric(sub['TRYGG_1'],     errors='coerce')
    trygg_total = pd.to_numeric(sub['TRYGG_total'], errors='coerce')

    # ── Housing ───────────────────────────────────────────────────────────
    boarea      = pd.to_numeric(sub['Boarea_Person'], errors='coerce')
    boarea_val  = boarea.dropna().iloc[0] if boarea.notna().any() else np.nan

    # ── Build feature dict ─────────────────────────────────────────────────
    feat: Dict = {
        # ─── Identifiers ─────────────────────────────────────────────────
        'household_id'           : household_id,
        'IndexDate_household'    : sub['IndexDate'].max(),
        'household_label'        : household_label,

        # ─── Case counts ─────────────────────────────────────────────────
        'household_size'         : household_size,
        'index_cases_count'      : index_cases_cnt,
        'secondary_cases_count'  : secondary_cases_cnt,

        # ─── Age (all members) ────────────────────────────────────────────
        'mean_age_2020'          : age.mean(),
        'max_age_2020'           : age.max(),
        'min_age_2020'           : age.min(),
        'age_variance'           : age.var(),
        'age_IQR'                : age.quantile(0.75) - age.quantile(0.25),
        'age_range'              : age.max() - age.min(),
        'age_0_17_count'         : int((age <= 17).sum()),
        'age_18_64_count'        : int(((age >= 18) & (age <= 64)).sum()),
        'age_65plus_count'       : int((age >= 65).sum()),
        'has_member_75plus'      : int((age >= 75).any()),
        'proportion_children'    : (age < 18).sum() / household_size,
        'proportion_elderly'     : (age >= 65).sum() / household_size,

        # ─── Immigration background ───────────────────────────────────────
        'prop_foreign_background': (sub['UtlSvBakg'] == 11).mean(),
        'has_any_foreign_background': int((sub['UtlSvBakg'] == 11).any()),
        'all_foreign_background' : int((sub['UtlSvBakg'] == 11).all()),
        'Fodelseland_diversity'  : sub['Fodelseland'].nunique(),
        'prop_born_sweden'       : (
            (sub['Fodelseland'] == 'SVERIGE').mean()
            if 'SVERIGE' in sub['Fodelseland'].values else 0.0
        ),

        # ─── Gender (all members) ─────────────────────────────────────────
        'male_count'             : male_count,
        'female_count'           : female_count,
        'proportion_male'        : male_count  / household_size,
        'proportion_female'      : female_count / household_size,
        'gender_diversity'       : int(male_count > 0 and female_count > 0),

        # ─── Family structure ─────────────────────────────────────────────
        'has_child_under_6'      : int((age < 6).any()),
        'has_child_6_17'         : int(((age >= 6) & (age <= 17)).any()),
        'has_elderly_65plus'     : int((age >= 65).any()),
        'multigenerational'      : int((age.max() - age.min()) > 40),
        'three_generation'       : int(
            (age < 18).any() and ((age >= 18) & (age <= 64)).any()
            and (age >= 65).any()
        ),

        # ─── Housing ──────────────────────────────────────────────────────
        'AntalBarnUnder18'       : (
            pd.to_numeric(sub['AntalBarnUnder18'], errors='coerce')
            .dropna().iloc[0]
            if pd.to_numeric(sub['AntalBarnUnder18'], errors='coerce').notna().any()
            else np.nan
        ),
        'Boarea_Person'          : boarea_val,
        'total_Boarea'           : boarea_val * household_size if pd.notna(boarea_val) else np.nan,
        'crowding_index'         : (
            household_size / boarea_val
            if pd.notna(boarea_val) and boarea_val > 0 else np.nan
        ),
        'is_overcrowded'         : int(
            household_size / boarea_val > 1.5
            if pd.notna(boarea_val) and boarea_val > 0 else 0
        ),
        'is_spacious'            : int(
            household_size / boarea_val < 0.5
            if pd.notna(boarea_val) and boarea_val > 0 else 0
        ),
        'Boendeform_mode'        : (
            sub['Boendeform'].mode()[0]
            if len(sub['Boendeform'].mode()) > 0 else np.nan
        ),

        # ─── Income (all members) ─────────────────────────────────────────
        'mean_DispInk04'         : disp_ink.mean(),
        'max_DispInk04'          : disp_ink.max(),
        'min_DispInk04'          : disp_ink.min(),
        'sd_DispInk04'           : disp_ink.std(),
        'median_DispInk04'       : disp_ink.median(),
        'range_DispInk04'        : disp_ink.max() - disp_ink.min(),
        'mean_DispInkFam04'      : disp_ink_fam.mean(),
        'max_DispInkFam04'       : disp_ink_fam.max(),
        'min_DispInkFam04'       : disp_ink_fam.min(),
        'sd_DispInkFam04'        : disp_ink_fam.std(),
        'median_DispInkFam04'    : disp_ink_fam.median(),
        'range_DispInkFam04'     : disp_ink_fam.max() - disp_ink_fam.min(),

        # ─── Homecare (all members) ───────────────────────────────────────
        'TRYGG_1_sum'            : trygg_1.sum(),
        'TRYGG_total_sum'        : trygg_total.sum(),
        'any_TRYGG_1'            : int(trygg_1.sum() > 0),
        'any_TRYGG'              : int(trygg_total.sum() > 0),
        'proportion_with_TRYGG'  : (trygg_total > 0).sum() / household_size,
        'TRYGG_total_per_capita' : trygg_total.sum() / household_size,
        'TRYGG_1_per_elderly'    : (
            trygg_1.sum() / int((age >= 65).sum())
            if int((age >= 65).sum()) > 0 else 0.0
        ),
    }

    # ── Index-case specific features [NEW] ────────────────────────────────
    index_sub = sub[sub['label'] == 1]

    if len(index_sub) == 0:
        # Fallback (should not reach here given validity check above)
        index_age        = pd.Series(dtype=float)
        index_income     = pd.Series(dtype=float)
        index_boarea     = pd.Series(dtype=float)
        index_trygg_tot  = pd.Series(dtype=float)
    else:
        i_birth_raw  = pd.to_numeric(index_sub['FodelseArMan'], errors='coerce')
        i_birth_year = (i_birth_raw // 100).astype('Int64')
        index_age    = (2020 - i_birth_year).astype('float64')
        index_income = pd.to_numeric(index_sub['DispInk04'],    errors='coerce')
        index_boarea = pd.to_numeric(index_sub['Boarea_Person'], errors='coerce')
        index_trygg_tot = pd.to_numeric(index_sub['TRYGG_total'], errors='coerce')

    index_kon     = pd.to_numeric(index_sub['Kon'], errors='coerce') \
                    if len(index_sub) > 0 else pd.Series(dtype=float)
    i_male        = int((index_kon == 1).sum()) if len(index_sub) > 0 else 0
    i_female      = int((index_kon == 2).sum()) if len(index_sub) > 0 else 0
    i_size        = len(index_sub)

    feat.update({
        # Who introduced the virus?
        'index_mean_age_2020'       : index_age.mean()    if i_size > 0 else np.nan,
        'index_min_age_2020'        : index_age.min()     if i_size > 0 else np.nan,
        'index_max_age_2020'        : index_age.max()     if i_size > 0 else np.nan,
        'index_has_elderly'         : int((index_age >= 65).any()) if i_size > 0 else 0,
        'index_has_child'           : int((index_age < 18).any())  if i_size > 0 else 0,
        'index_proportion_female'   : (i_female / i_size) if i_size > 0 else np.nan,
        'index_proportion_foreign'  : (
            (index_sub['UtlSvBakg'] == 11).mean() if i_size > 0 else np.nan
        ),
        'index_mean_DispInk04'      : index_income.mean()   if i_size > 0 else np.nan,
        'index_mean_Boarea'         : index_boarea.mean()   if i_size > 0 else np.nan,
        'index_any_TRYGG'           : (
            int(index_trygg_tot.sum() > 0) if i_size > 0 else 0
        ),
        'index_mean_TRYGG_total'    : (
            index_trygg_tot.mean() if i_size > 0 else np.nan
        ),
        # Ratio: index cases relative to household size
        'index_to_household_ratio'  : i_size / household_size,
    })

    return feat

def aggregate_to_household_level(person_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load household membership map and aggregate all household features.

    Returns a DataFrame with one row per household.
    """
    print(f"\n{'='*70}")
    print("STEP 2 – AGGREGATING TO HOUSEHOLD LEVEL")
    print(f"{'='*70}")

    # Load household mapping
    print(f"  Loading household map: {Config.HOUSEHOLD_MAP_PATH}")
    hh_map = pd.read_csv(
        Config.HOUSEHOLD_MAP_PATH,
        encoding=Config.ENCODING,
        low_memory=False,
    )
    hh_id_col   = hh_map.columns[0]
    member_cols = [c for c in hh_map.columns if c.startswith('member_')]

    total_hh = len(hh_map)
    print(f"  Total households in map : {total_hh:,}")
    print(f"  Member columns          : {len(member_cols)}")

    if Config.TEST_MODE:
        hh_map    = hh_map.head(Config.TEST_HOUSEHOLDS)
        total_hh  = len(hh_map)
        print(f"  ⚠ TEST MODE: limited to {total_hh:,} households")

    all_rows = []
    n_batches = (total_hh + Config.AGGREGATE_BATCH - 1) // Config.AGGREGATE_BATCH

    for b in range(n_batches):
        start = b * Config.AGGREGATE_BATCH
        end   = min(start + Config.AGGREGATE_BATCH, total_hh)
        batch = hh_map.iloc[start:end]

        print(f"\n  Batch {b+1}/{n_batches}: households {start:,}–{end-1:,}")

        for _, row in tqdm(batch.iterrows(), total=len(batch),
                           desc=f"  Batch {b+1}"):
            hid     = str(row.iloc[0])
            members = (
                row[member_cols]
                .dropna()
                .astype(str)
                .str.rstrip('.0')
                .tolist()
            )
            feat = compute_household_features(hid, members, person_df)
            if feat is not None:
                all_rows.append(feat)

        gc.collect()

    hh_df = pd.DataFrame(all_rows)

    label_dist = hh_df['household_label'].value_counts().sort_index().to_dict()
    print(f"\n  Valid households  : {len(hh_df):,}")
    print(f"  Label distribution: {label_dist}")
    print(f"  Transmission rate : {hh_df['household_label'].mean()*100:.1f}%")

    return hh_df

# STEP 3 – ENCODE AND IMPUTE

def encode_and_impute(
    df: pd.DataFrame,
    mode: str = 'fit',
    thresholds: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """
    Encode categorical features and impute missing values.

    Encoding strategy
    -----------------
    Boendeform_mode → OneHotEncoding
    Fodelseland_mode (if present) → frequency encoding
    All numeric columns → median imputation

    Parameters
    ----------
    mode       : 'fit' or 'transform'
    thresholds : dict of fitted values (required for 'transform')
    """
    if mode == 'fit':
        thresholds = {}

    # ── Drop high-missing columns (fit only) ─────────────────────────────
    if mode == 'fit':
        miss_rates = df.isnull().mean() * 100
        drop_cols  = miss_rates[miss_rates > Config.MISSING_RATE_DROP].index.tolist()
        # Never drop identifiers / labels
        drop_cols  = [c for c in drop_cols
                      if c not in ('household_id', 'household_label',
                                   'IndexDate_household')]
        thresholds['drop_cols'] = drop_cols
        if drop_cols:
            print(f"  Dropping {len(drop_cols)} columns with >{Config.MISSING_RATE_DROP}% missing:")
            for c in drop_cols:
                print(f"    {c} ({miss_rates[c]:.1f}%)")
    else:
        drop_cols = thresholds.get('drop_cols', [])

    df.drop(columns=drop_cols, errors='ignore', inplace=True)

    # ── Boendeform_mode – OHE ────────────────────────────────────────────
    if 'Boendeform_mode' in df.columns:
        df['Boendeform_mode'] = df['Boendeform_mode'].fillna('unknown').astype(str)
        if mode == 'fit':
            ohe      = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            enc_arr  = ohe.fit_transform(df[['Boendeform_mode']])
            thresholds['boende_cats'] = list(ohe.categories_[0])
        else:
            cats    = thresholds['boende_cats']
            ohe     = OneHotEncoder(sparse_output=False,
                                    categories=[cats],
                                    handle_unknown='ignore')
            enc_arr = ohe.fit_transform(df[['Boendeform_mode']])
        boende_cols = [f'Boendeform_{c}' for c in (
            thresholds['boende_cats'] if mode == 'transform'
            else ohe.categories_[0]
        )]
        df = pd.concat(
            [df.drop(columns=['Boendeform_mode']),
             pd.DataFrame(enc_arr, columns=boende_cols, index=df.index)],
            axis=1,
        )

    # ── Numeric columns – median imputation ──────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    skip_cols    = {'household_label'}
    for col in numeric_cols:
        if col in skip_cols:
            continue
        if df[col].isna().any():
            if mode == 'fit':
                med = df[col].median()
                thresholds[f'{col}_median'] = med
            else:
                med = thresholds.get(f'{col}_median', df[col].median())
            df[col] = df[col].fillna(med)

    # ── Log-transform skewed financial / area features ────────────────────
    log_cols = [
        'Boarea_Person', 'total_Boarea', 'crowding_index',
        'mean_DispInk04', 'mean_DispInkFam04',
        'index_mean_DispInk04', 'index_mean_Boarea',
    ]
    for col in log_cols:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))

    return df, thresholds

# STEP 4 – STANDARDISE

def standardise(
    train: pd.DataFrame,
    val:   Optional[pd.DataFrame],
    test:  Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Fit StandardScaler on training set, apply to val and test.
    Binary, count, and categorical columns are left unchanged.
    """
    skip_suffixes = ('_label', '_id', '_mode', 'IndexDate_household')
    skip_prefixes = ('Boendeform_', 'is_', 'has_', 'any_',
                     'three_', 'multi', 'gender_')
    skip_exact    = {'household_size', 'index_cases_count',
                     'secondary_cases_count', 'household_label'}

    cols_to_scale = [
        c for c in train.columns
        if c not in skip_exact
        and train[c].dtype in (np.float32, np.float64, float, np.float16)
        and not any(c.endswith(s) for s in skip_suffixes)
        and not any(c.startswith(s) for s in skip_prefixes)
    ]

    print(f"\n  Standardising {len(cols_to_scale)} continuous features …")

    for col in tqdm(cols_to_scale, desc="  Scaling"):
        scaler     = StandardScaler()
        train_vals = np.nan_to_num(
            train[[col]].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0
        )
        scaler.fit(train_vals)
        train[col + '_std'] = scaler.transform(train_vals).flatten()
        train.drop(columns=[col], inplace=True)

        for split in [val, test]:
            if split is not None and col in split.columns:
                vals = np.nan_to_num(
                    split[[col]].values.astype(float), nan=0.0,
                    posinf=0.0, neginf=0.0
                )
                split[col + '_std'] = scaler.transform(vals).flatten()
                split.drop(columns=[col], inplace=True)

    gc.collect()
    return train, val, test

# STEP 5 – K-FOLD SPLIT AND EXPORT

def create_kfold_datasets(hh_df: pd.DataFrame) -> None:
    """
    Stratified K-fold split → encode → standardise → save.
    """
    print(f"\n{'='*70}")
    print(f"STEP 5 – K-FOLD DATASET CREATION ({Config.N_FOLDS} folds)")
    print(f"{'='*70}")

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # ── Hold-out test split ───────────────────────────────────────────────
    train_full, test_df = train_test_split(
        hh_df,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=hh_df['household_label'],
    )
    del hh_df
    gc.collect()

    print(f"  Train pool : {len(train_full):,}")
    print(f"  Test set   : {len(test_df):,}")

    skf   = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE
    )
    folds = list(skf.split(train_full, train_full['household_label']))
    fold_info_lines = [
        f"Household-Level Dataset V2 — K-Fold Info",
        "=" * 50,
        f"N folds      : {Config.N_FOLDS}",
        f"Random state : {Config.RANDOM_STATE}",
        f"Test size    : {Config.TEST_SIZE*100:.0f}%",
        f"Train pool   : {len(train_full):,}",
        f"Test set     : {len(test_df):,}",
        "",
    ]

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        fold_num = fold_idx + 1
        print(f"\n{'='*70}")
        print(f"  FOLD {fold_num}/{Config.N_FOLDS}")
        print(f"{'='*70}")

        train_fold = train_full.iloc[tr_idx].copy()
        val_fold   = train_full.iloc[val_idx].copy()

        print(f"  Train: {len(train_fold):,} | Val: {len(val_fold):,}")

        # Encode & impute (fit on train only)
        train_fold, thresholds = encode_and_impute(train_fold, mode='fit')
        val_fold,   _          = encode_and_impute(val_fold,   mode='transform',
                                                    thresholds=thresholds)
        test_copy = test_df.copy()
        test_copy, _           = encode_and_impute(test_copy,  mode='transform',
                                                    thresholds=thresholds)

        # Align columns (transform may not see all categories)
        all_cols = sorted(set(train_fold.columns) &
                          set(val_fold.columns)   &
                          set(test_copy.columns))
        train_fold = train_fold[all_cols]
        val_fold   = val_fold[all_cols]
        test_copy  = test_copy[all_cols]

        # Standardise
        train_fold, val_fold, test_copy = standardise(
            train_fold, val_fold, test_copy
        )

        # Align again after standardisation
        all_cols2 = sorted(set(train_fold.columns) &
                           set(val_fold.columns)   &
                           set(test_copy.columns))
        train_fold = train_fold[all_cols2]
        val_fold   = val_fold[all_cols2]
        test_copy  = test_copy[all_cols2]

        train_path = os.path.join(Config.OUTPUT_DIR, f'train_fold_{fold_num}.csv')
        val_path   = os.path.join(Config.OUTPUT_DIR, f'val_fold_{fold_num}.csv')
        test_path  = os.path.join(Config.OUTPUT_DIR, f'test_fold_{fold_num}.csv')

        train_fold.to_csv(train_path, index=False, encoding=Config.ENCODING)
        val_fold.to_csv(val_path,     index=False, encoding=Config.ENCODING)
        test_copy.to_csv(test_path,   index=False, encoding=Config.ENCODING)

        print(f"  Saved → {train_path}")
        print(f"  Final columns: {train_fold.shape[1]}")

        fold_info_lines += [
            f"Fold {fold_num}:",
            f"  Train : {len(train_fold):,} | Val : {len(val_fold):,}",
            f"  Columns: {train_fold.shape[1]}",
            f"  Label dist (train): "
            f"{train_fold['household_label'].value_counts().sort_index().to_dict()}",
            "",
        ]

        del train_fold, val_fold, test_copy, thresholds
        gc.collect()

    info_path = os.path.join(Config.OUTPUT_DIR, 'folds_info.txt')
    with open(info_path, 'w') as f:
        f.write('\n'.join(fold_info_lines))
    print(f"\n  Fold info → {info_path}")

# MAIN PIPELINE

def run_household_v2_pipeline() -> None:
    """Execute the full household-level V2 dataset creation pipeline."""
    print("=" * 70)
    print("HOUSEHOLD-LEVEL DATASET CREATOR (V2 — No Count Features)")
    print("COVID-19 Within-Household Transmission — Binary Classification")
    print("=" * 70)

    # ── Step 1: Load person table (base cols only) ─────────────────────────
    person_df = load_person_table()

    # ── Step 2: Aggregate to household level ──────────────────────────────
    hh_df = aggregate_to_household_level(person_df)
    del person_df
    gc.collect()

    # ── Step 3-5: Encode, standardise, K-fold export ─────────────────────
    create_kfold_datasets(hh_df)

    print("\n" + "=" * 70)
    print("HOUSEHOLD-LEVEL V2 PIPELINE COMPLETE")
    print(f"Output directory: {Config.OUTPUT_DIR}/")
    print("=" * 70)

if __name__ == '__main__':
    run_household_v2_pipeline()
