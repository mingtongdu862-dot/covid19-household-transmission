"""
Personal-Level Dataset Creator — Framework II (Marginal Individual Risk)
With Household-Level Aggregated Features
=========================================================================

Probabilistic framework
------------------------
Framework II estimates the marginal individual infection probability:

    P(Y_i = 1 | X_i, Z_h)

where X_i = individual features and Z_h = household-level features.
Unlike Framework I, this model operates as a stand-alone estimator and
must NOT be multiplied by the household model output.

Two feature layers
------------------
1. Individual features X_i  (same as Framework I / V2):
     Core socioeconomic / demographic + medical code counts (two-stage
     chi-square + epidemiological whitelist filter).

2. Household features Z_h  (NEW — not in Framework I):
     Aggregated statistics computed from all household members (including
     index cases), joined to every individual row so that members of the
     same household share identical Z_h values.
     All household columns are prefixed with 'hh_' to distinguish them
     from individual features.

     This is the natural addition for Framework II: since the individual
     model estimates P(Y_i=1 | X_i, Z_h) directly, including Z_h gives
     the model both who the person is and what household environment they
     face, simultaneously.

3. T_h  (binary household transmission indicator):
     1 if the household had ≥1 secondary case, 0 otherwise.
     Treated as a core feature (bypasses chi-square filtering).

Why a separate person table load for household features
--------------------------------------------------------
_compute_hh_features_for_one() uses ALL household members INCLUDING
index cases (original label=1) to compute statistics such as
index_mean_age_2020, index_proportion_female, etc.
However, load_and_relabel() drops all index cases before ML labelling.
A second lightweight load of the person table (base columns only,
original labels preserved) is therefore performed specifically for
household aggregation, then discarded.

Label definition
-----------------
  original 0 → 0   healthy member (not infected)
  original 1 → X   DROPPED (co-primary / index case, not a target)
  original 2 → 1   secondary case (positive class)

Negative class (Framework II):
  label=0 members from BOTH T_h=0 AND T_h=1 households are retained.

Input
-----
  Feature_Tables/Raw_Feature_Secondary_Case.csv   person-level features
  Feature_Tables/household_member.csv             household membership map
  Features_Selected_Data/HushallPerson_2019_duplicates.pkl

Output
------
  Personal_Level_Dataset_FrameworkII/
    train_fold_{1..N}.csv
    val_fold_{1..N}.csv
    test_fold_{1..N}.csv
    feature_selection_report.txt

Author : [Your Name]
Date   : 2025-01-31
"""

import os
import gc
import warnings
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_selection import chi2
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings('ignore')

# CONFIGURATION
class Config:

    # I/O paths
    PERSON_TABLE_PATH  = 'Feature_Tables/Raw_Feature_Secondary_Case.csv'
    HOUSEHOLD_MAP_PATH = 'Feature_Tables/household_member.csv'   # NEW
    OUTPUT_DIR         = 'Individual_Level_Dataset_FrameworkII'
    ENCODING           = 'latin1'

    # Person -> household ID mapping (from preprocessing pipeline)
    HUSHALL_PKL_PATH   = 'Features_Selected_Data/HushallPerson_2019_duplicates.pkl'

    # Reading
    PERSON_CHUNK_SIZE  = 500_000
    AGGREGATE_BATCH    = 100_000   # households per aggregation batch

    # Household feature imputation
    HH_MISSING_RATE_DROP = 20     # drop hh_ columns with >20% missing

    # Feature selection (count features only)
    CHI2_P_THRESHOLD  = 0.05
    MAX_FREQ_THRESH   = 0.95
    CHI2_BATCH_SIZE   = 2_000

    # K-fold
    TEST_SIZE    = 0.10
    N_FOLDS      = 5
    RANDOM_STATE = 42

    # Core individual features — always retained, bypass chi-square.
    # T_h is the household transmission indicator (Framework II).
    CORE_FEATURES = [
        'UtlSvBakg', 'Fodelseland', 'FodelseArMan', 'Kon',
        'AntalBarnUnder18', 'Boarea_Person', 'Boendeform',
        'DispInk04', 'DispInkFam04', 'TRYGG_1', 'TRYGG_total',
        'T_h',
    ]

    # Base person columns to load for household aggregation (no count cols)
    PERSON_BASE_COLS = [
        'person_id', 'IndexDate', 'label',
        'UtlSvBakg', 'Fodelseland', 'FodelseArMan', 'Kon',
        'AntalBarnUnder18', 'Boarea_Person', 'Boendeform',
        'DispInk04', 'DispInkFam04', 'TRYGG_1', 'TRYGG_total',
    ]

    # Count-feature column prefixes
    COUNT_PREFIXES = ('contact_', 'lmed_', 'ov_', 'sv_')

    # Household feature columns to exclude from the join to person rows
    # (leaky or already handled as T_h).
    # NOTE: 'household_id' is intentionally NOT listed here — it must be
    # retained in hh_for_join as the merge key and is dropped only after
    # the join completes (see attach_household_features step 6).
    HH_EXCLUDE_COLS = frozenset({
        'IndexDate_household',
        'household_label',       # = T_h, added separately
        'secondary_cases_count', # directly encodes label -> leaky
    })

    # Prefix applied to all joined household features
    HH_PREFIX = 'hh_'

# EPIDEMIOLOGICAL RELEVANCE WHITELISTS  (identical to V2)

RELEVANT_ATC_PREFIXES: Tuple[str, ...] = (
    'J01', 'J02', 'J04', 'J05', 'J06', 'J07',
    'R01', 'R02', 'R03', 'R05', 'R06', 'R07',
    'L01', 'L02', 'L03', 'L04',
    'H02', 'H03',
    'A10',
    'B01',
    'C07', 'C08', 'C09', 'C10',
    'M01', 'N02',
    'N05', 'N06',
)

RELEVANT_ICD_PREFIXES: Tuple[str, ...] = (
    'A', 'B',
    'C', 'D3', 'D4',
    'D5', 'D6', 'D7', 'D8',
    'E10', 'E11', 'E12', 'E13', 'E14',
    'E65', 'E66', 'E67', 'E68',
    'F',
    'G0',
    'I1', 'I2', 'I26', 'I4', 'I5', 'I6', 'I7', 'I8',
    'J',
    'K5', 'K7',
    'M0', 'M3',
    'N17', 'N18', 'N19',
    'O',
    'U07', 'U08', 'U09',
    'Z2',
)

RELEVANT_CONTACT_KEYWORDS: Tuple[str, ...] = (
    'ndningsbesv', 'Hosta', 'Heshet', 'heshet', 'Bih', 'Feber',
    'alsont', 'Halsont', 'snuva', 'Näst', 'Luktf', 'Smakf',
    'iarr', 'llam', 'räkning', 'Kräkning', 'Utslag', 'välj',
    'Trötthet', 'tötthet', 'Tr\xe6tthet',
    'llergisk', 'Allergi', 'Klåda', 'Kl\xe5da',
    'nfektionstecken', 'Bett av', 'Mask',
    'Blodsocker', 'Blodtryck', 'Rytm',
    'Urinv', 'Yrsel', 'Buksm', 'Svettning', 'Förvirring',
    'Medvetande', 'Skrakningar', 'Skakningar', 'Kramp',
    'Nedst', 'Oro', 'ngest',
    'Hudf', 'ld', 'lösor',
    'Bröstsmärta', 'Br\xf6stsmärta', 'Blodig upphostning',
    'Nedsatt aptit', 'Viktnedg',
)

# UTILITIES

def force_cleanup(*objects) -> None:
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()

def is_atc_relevant(col: str) -> bool:
    code = col[len('lmed_'):]
    return any(code.startswith(p) for p in RELEVANT_ATC_PREFIXES)

def is_icd_relevant(col: str) -> bool:
    code = col[3:]
    return any(code.startswith(p) for p in RELEVANT_ICD_PREFIXES)

def is_contact_relevant(col: str) -> bool:
    symptom = col[8:]
    return any(kw in symptom for kw in RELEVANT_CONTACT_KEYWORDS)

def apply_epidemiological_filter(
        columns: List[str]) -> Tuple[List[str], Dict[str, int]]:
    kept = []
    stats: Dict[str, int] = {
        'lmed_kept': 0, 'lmed_total': 0,
        'ov_kept':   0, 'ov_total':   0,
        'sv_kept':   0, 'sv_total':   0,
        'contact_kept': 0, 'contact_total': 0,
    }
    for col in columns:
        if col.startswith('lmed_'):
            stats['lmed_total'] += 1
            if is_atc_relevant(col):
                kept.append(col); stats['lmed_kept'] += 1
        elif col.startswith('ov_'):
            stats['ov_total'] += 1
            if is_icd_relevant(col):
                kept.append(col); stats['ov_kept'] += 1
        elif col.startswith('sv_'):
            stats['sv_total'] += 1
            if is_icd_relevant(col):
                kept.append(col); stats['sv_kept'] += 1
        elif col.startswith('contact_'):
            stats['contact_total'] += 1
            if is_contact_relevant(col):
                kept.append(col); stats['contact_kept'] += 1
    return kept, stats

# STEP 1 – LOAD AND RELABEL  (unchanged from Framework II base script)

def load_and_relabel(path: str, encoding: str, chunk_size: int) -> pd.DataFrame:
    """
    Load Raw_Feature_Secondary_Case.csv and apply label scheme:
        original 0 -> 0  (healthy member; T_h=0 households retained)
        original 1 -> X  DROPPED (co-primary / index case)
        original 2 -> 1  (secondary case, positive class)
    """
    print(f"\n{'='*70}")
    print("STEP 1 – LOADING, FILTERING AND RELABELLING  [Framework II]")
    print(f"  Source : {path}")
    print(f"  Scheme : label=2->1  |  label=0->0  |  label=1 EXCLUDED")
    print(f"  NOTE   : T_h=0 healthy members RETAINED (Framework II)")
    print(f"{'='*70}")

    chunks    = []
    n_total   = 0
    n_dropped = 0

    for i, chunk in enumerate(
        pd.read_csv(path, chunksize=chunk_size, encoding=encoding,
                    low_memory=False)
    ):
        chunk['person_id'] = chunk['person_id'].astype(str).str.rstrip('.0')
        chunk['label']     = pd.to_numeric(chunk['label'], errors='coerce')
        n_total   += len(chunk)
        n_dropped += (chunk['label'] == 1).sum()
        chunk      = chunk[chunk['label'] != 1].copy()
        chunk['label'] = chunk['label'].apply(lambda x: 1 if x == 2 else x)
        chunks.append(chunk)
        print(f"  Chunk {i+1}: {len(chunk):,} rows  "
              f"(secondary: {(chunk['label']==1).sum():,} | "
              f"healthy: {(chunk['label']==0).sum():,})")

    df = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()

    label_dist = df['label'].value_counts().sort_index().to_dict()
    print(f"\n  Total in CSV   : {n_total:,}")
    print(f"  Dropped (lbl=1): {n_dropped:,} ({n_dropped/n_total*100:.1f}%)")
    print(f"  Remaining      : {len(df):,}  dist={label_dist}")
    return df

# STEP 1b – SEPARATE PERSON TABLE LOAD FOR HOUSEHOLD AGGREGATION

def load_person_table_for_hh_features() -> pd.DataFrame:
    """
    Load the person-level table keeping ORIGINAL labels (0/1/2), base columns
    only (no count features), indexed by person_id.

    This separate load is required because _compute_hh_features_for_one()
    identifies index cases via label==1 (original) to compute index-case
    sub-statistics (index_mean_age_2020, index_proportion_female, etc.),
    but those rows are dropped in load_and_relabel().
    """
    print("  Loading person table with original labels for Z_h computation …")
    header   = pd.read_csv(Config.PERSON_TABLE_PATH, nrows=0,
                            encoding=Config.ENCODING, low_memory=False)
    use_cols = [c for c in Config.PERSON_BASE_COLS if c in header.columns]
    del header; gc.collect()

    chunks = []
    for chunk in pd.read_csv(
        Config.PERSON_TABLE_PATH, usecols=use_cols,
        chunksize=Config.PERSON_CHUNK_SIZE,
        encoding=Config.ENCODING, low_memory=False,
    ):
        chunk['person_id'] = chunk['person_id'].astype(str).str.rstrip('.0')
        chunk.set_index('person_id', inplace=True)
        chunk['label'] = pd.to_numeric(chunk['label'], errors='coerce')
        chunks.append(chunk)

    person_df = pd.concat(chunks, ignore_index=False)
    del chunks; gc.collect()
    person_df = person_df[~person_df.index.duplicated(keep='first')]
    print(f"  Person table: {len(person_df):,} rows "
          f"(label dist: "
          f"{person_df['label'].value_counts().sort_index().to_dict()})")
    return person_df

# STEP 1.5 – COMPUTE Z_h AND ANNOTATE T_h

def _compute_hh_features_for_one(
    household_id: str,
    members: List[str],
    person_df: pd.DataFrame,
) -> Optional[Dict]:
    """
    Compute household-level aggregated features for a single household.
    Mirrors compute_household_features() from Household_Level_Dataset_Creator.

    Validity: needs >=1 index case AND >=1 non-index member.
    Returns None for invalid / pure-index households.
    """
    valid  = [m for m in members if m in person_df.index]
    if not valid:
        return None

    sub                 = person_df.loc[valid]
    index_cnt           = int((sub['label'] == 1).sum())
    secondary_cnt       = int((sub['label'] == 2).sum())
    hh_size             = len(sub)

    if index_cnt == 0:
        return None
    if secondary_cnt == 0 and hh_size == index_cnt:
        return None   # pure-index household

    # ── Age ─────────────────────────────────────────────────────────────────
    birth = pd.to_numeric(sub['FodelseArMan'], errors='coerce')
    age   = (2020 - (birth // 100)).astype('float64')

    # ── Gender ──────────────────────────────────────────────────────────────
    kon     = pd.to_numeric(sub['Kon'], errors='coerce')
    n_male  = int((kon == 1).sum())
    n_female= int((kon == 2).sum())

    # ── Income ──────────────────────────────────────────────────────────────
    inc     = pd.to_numeric(sub['DispInk04'],    errors='coerce')
    inc_fam = pd.to_numeric(sub['DispInkFam04'], errors='coerce')

    # ── Homecare ────────────────────────────────────────────────────────────
    tr1 = pd.to_numeric(sub['TRYGG_1'],     errors='coerce')
    trt = pd.to_numeric(sub['TRYGG_total'], errors='coerce')

    # ── Housing ─────────────────────────────────────────────────────────────
    boa     = pd.to_numeric(sub['Boarea_Person'], errors='coerce')
    boa_val = boa.dropna().iloc[0] if boa.notna().any() else np.nan

    feat: Dict = {
        'household_id'              : household_id,
        'IndexDate_household'       : (sub['IndexDate'].max()
                                        if 'IndexDate' in sub.columns
                                        else np.nan),
        'household_label'           : 1 if secondary_cnt > 0 else 0,
        'secondary_cases_count'     : secondary_cnt,
        'household_size'            : hh_size,
        'index_cases_count'         : index_cnt,
        # Age
        'mean_age_2020'             : age.mean(),
        'max_age_2020'              : age.max(),
        'min_age_2020'              : age.min(),
        'age_variance'              : age.var(),
        'age_IQR'                   : age.quantile(0.75) - age.quantile(0.25),
        'age_range'                 : age.max() - age.min(),
        'age_0_17_count'            : int((age <= 17).sum()),
        'age_18_64_count'           : int(((age >= 18) & (age <= 64)).sum()),
        'age_65plus_count'          : int((age >= 65).sum()),
        'has_member_75plus'         : int((age >= 75).any()),
        'proportion_children'       : (age < 18).sum() / hh_size,
        'proportion_elderly'        : (age >= 65).sum() / hh_size,
        # Immigration background
        'prop_foreign_background'   : (sub['UtlSvBakg'] == 11).mean(),
        'has_any_foreign_background': int((sub['UtlSvBakg'] == 11).any()),
        'all_foreign_background'    : int((sub['UtlSvBakg'] == 11).all()),
        'Fodelseland_diversity'     : sub['Fodelseland'].nunique(),
        'prop_born_sweden'          : (
            (sub['Fodelseland'] == 'SVERIGE').mean()
            if 'SVERIGE' in sub['Fodelseland'].values else 0.0),
        # Gender
        'male_count'                : n_male,
        'female_count'              : n_female,
        'proportion_male'           : n_male  / hh_size,
        'proportion_female'         : n_female / hh_size,
        'gender_diversity'          : int(n_male > 0 and n_female > 0),
        # Family structure
        'has_child_under_6'         : int((age < 6).any()),
        'has_child_6_17'            : int(((age >= 6) & (age <= 17)).any()),
        'has_elderly_65plus'        : int((age >= 65).any()),
        'multigenerational'         : int((age.max() - age.min()) > 40),
        'three_generation'          : int(
            (age < 18).any() and ((age >= 18) & (age <= 64)).any()
            and (age >= 65).any()),
        # Housing — only genuinely aggregated / derived quantities.
        # Excluded (identical to individual X_i features, not aggregated):
        #   AntalBarnUnder18  -> same value for all household members
        #   Boarea_Person     -> same value for all household members
        #   Boendeform_mode   -> same dwelling type as individual Boendeform
        'total_Boarea'              : (boa_val * hh_size
                                        if pd.notna(boa_val) else np.nan),
        'crowding_index'            : (
            hh_size / boa_val
            if pd.notna(boa_val) and boa_val > 0 else np.nan),
        'is_overcrowded'            : int(
            hh_size / boa_val > 1.5
            if pd.notna(boa_val) and boa_val > 0 else 0),
        'is_spacious'               : int(
            hh_size / boa_val < 0.5
            if pd.notna(boa_val) and boa_val > 0 else 0),
        # Income
        'mean_DispInk04'            : inc.mean(),
        'max_DispInk04'             : inc.max(),
        'min_DispInk04'             : inc.min(),
        'sd_DispInk04'              : inc.std(),
        'median_DispInk04'          : inc.median(),
        'range_DispInk04'           : inc.max() - inc.min(),
        'mean_DispInkFam04'         : inc_fam.mean(),
        'max_DispInkFam04'          : inc_fam.max(),
        'min_DispInkFam04'          : inc_fam.min(),
        'sd_DispInkFam04'           : inc_fam.std(),
        'median_DispInkFam04'       : inc_fam.median(),
        'range_DispInkFam04'        : inc_fam.max() - inc_fam.min(),
        # Homecare
        'TRYGG_1_sum'               : tr1.sum(),
        'TRYGG_total_sum'           : trt.sum(),
        'any_TRYGG_1'               : int(tr1.sum() > 0),
        'any_TRYGG'                 : int(trt.sum() > 0),
        'proportion_with_TRYGG'     : (trt > 0).sum() / hh_size,
        'TRYGG_total_per_capita'    : trt.sum() / hh_size,
        'TRYGG_1_per_elderly'       : (
            tr1.sum() / int((age >= 65).sum())
            if int((age >= 65).sum()) > 0 else 0.0),
    }

    # ── Index-case features ─────────────────────────────────────────────────
    idx  = sub[sub['label'] == 1]
    isz  = len(idx)
    if isz > 0:
        ib   = pd.to_numeric(idx['FodelseArMan'], errors='coerce')
        iage = (2020 - (ib // 100)).astype('float64')
        iinc = pd.to_numeric(idx['DispInk04'],    errors='coerce')
        iboa = pd.to_numeric(idx['Boarea_Person'], errors='coerce')
        itrg = pd.to_numeric(idx['TRYGG_total'],   errors='coerce')
        ikon = pd.to_numeric(idx['Kon'],           errors='coerce')
        ifem = int((ikon == 2).sum())
    else:
        iage = iinc = iboa = itrg = pd.Series(dtype=float)
        ifem = 0

    feat.update({
        'index_mean_age_2020'       : iage.mean()           if isz > 0 else np.nan,
        'index_min_age_2020'        : iage.min()            if isz > 0 else np.nan,
        'index_max_age_2020'        : iage.max()            if isz > 0 else np.nan,
        'index_has_elderly'         : int((iage >= 65).any()) if isz > 0 else 0,
        'index_has_child'           : int((iage < 18).any())  if isz > 0 else 0,
        'index_proportion_female'   : (ifem / isz)          if isz > 0 else np.nan,
        'index_proportion_foreign'  : (
            (idx['UtlSvBakg'] == 11).mean() if isz > 0 else np.nan),
        'index_mean_DispInk04'      : iinc.mean()           if isz > 0 else np.nan,
        'index_mean_Boarea'         : iboa.mean()           if isz > 0 else np.nan,
        'index_any_TRYGG'           : int(itrg.sum() > 0)   if isz > 0 else 0,
        'index_mean_TRYGG_total'    : itrg.mean()           if isz > 0 else np.nan,
        'index_to_household_ratio'  : isz / hh_size,
    })

    return feat

def attach_household_features(
    df: pd.DataFrame,
    person_df_full: pd.DataFrame,
    hushall_dict: Dict,
) -> pd.DataFrame:
    """
    For each individual in df:
      1. Map person_id -> household_id via hushall_dict.
      2. Compute Z_h (all household-level aggregated features) for each
         household using person_df_full (original labels).
      3. Annotate T_h (1 if the household had >=1 secondary case, 0 otherwise).
      4. Join all Z_h features to df with 'hh_' prefix.
         Members of the same household receive IDENTICAL Z_h values.

    Parameters
    ----------
    df             : individual DataFrame after load_and_relabel()
    person_df_full : person table with ORIGINAL labels, indexed by person_id
    hushall_dict   : {person_id: [{P1105_LopNr_Hushallsid_2019: hid, ...}]}

    Returns
    -------
    df with new columns: 'T_h' and all 'hh_*' household feature columns.
    """
    print(f"\n{'='*70}")
    print("STEP 1.5 – COMPUTING Z_h AND ANNOTATING T_h")
    print(f"  Household map  : {Config.HOUSEHOLD_MAP_PATH}")
    print(f"  hh_ prefix applied to all household features")
    print(f"{'='*70}")

    # ── 1. Map every person -> household_id ──────────────────────────────────
    print("  Mapping persons -> household IDs …")

    def get_hid(pid: str) -> Optional[str]:
        entry = hushall_dict.get(pid)
        if entry and isinstance(entry, list) and len(entry) > 0:
            hid = str(
                entry[0].get('P1105_LopNr_Hushallsid_2019', '')
            ).rstrip('.0')
            return hid if hid else None
        return None

    df['_household_id'] = df['person_id'].map(get_hid)
    n_unmapped = df['_household_id'].isna().sum()
    if n_unmapped > 0:
        print(f"  WARNING: {n_unmapped:,} persons without household mapping "
              f"-> T_h=0, hh_ features=NaN")

    # ── 2. Load household membership map ────────────────────────────────────
    print(f"  Loading household membership map …")
    hh_map      = pd.read_csv(Config.HOUSEHOLD_MAP_PATH,
                               encoding=Config.ENCODING, low_memory=False)
    member_cols = [c for c in hh_map.columns if c.startswith('member_')]
    total_hh    = len(hh_map)
    print(f"  Households in map: {total_hh:,}")

    # ── 3. Compute Z_h for all households (batched) ──────────────────────────
    print(f"  Computing Z_h features (batched) …")
    all_hh_rows = []
    n_batches   = (total_hh + Config.AGGREGATE_BATCH - 1) // Config.AGGREGATE_BATCH

    for b in range(n_batches):
        start = b * Config.AGGREGATE_BATCH
        end   = min(start + Config.AGGREGATE_BATCH, total_hh)
        batch = hh_map.iloc[start:end]
        print(f"  Batch {b+1}/{n_batches}: households {start:,}–{end-1:,}")

        for _, row in tqdm(batch.iterrows(), total=len(batch),
                           desc=f"  Batch {b+1}"):
            hid     = str(row.iloc[0]).rstrip('.0')  # normalise: '123.0' -> '123'
            members = (row[member_cols].dropna().astype(str)
                                       .str.rstrip('.0').tolist())
            feat = _compute_hh_features_for_one(hid, members, person_df_full)
            if feat is not None:
                all_hh_rows.append(feat)

        gc.collect()

    hh_feat_df = pd.DataFrame(all_hh_rows)
    del all_hh_rows; gc.collect()

    print(f"  Valid households computed : {len(hh_feat_df):,}")
    print(f"  Raw Z_h columns           : {len(hh_feat_df.columns)}")

    # ── 4. T_h: identify transmission households ─────────────────────────────
    transmission_hids = set(
        hh_feat_df.loc[hh_feat_df['household_label'] == 1, 'household_id']
        .astype(str).unique()
    )
    print(f"  T_h=1 households: {len(transmission_hids):,}")

    df['T_h'] = df['_household_id'].apply(
        lambda hid: (1 if hid in transmission_hids
                     else (0 if pd.notna(hid) else np.nan))
    ).astype('Int8')

    # ── 5. Build Z_h lookup table with hh_ prefix ────────────────────────────
    # Drop leaky / duplicate columns (HH_EXCLUDE_COLS); household_id is
    # kept as the merge key and excluded from renaming / scaling.
    drop_cols   = list(Config.HH_EXCLUDE_COLS & set(hh_feat_df.columns))
    hh_for_join = hh_feat_df.drop(columns=drop_cols, errors='ignore').copy()

    # High-missing column removal — evaluate only feature columns,
    # not the merge key (household_id).
    feature_cols_for_miss = [c for c in hh_for_join.columns
                              if c != 'household_id']
    miss_rates = hh_for_join[feature_cols_for_miss].isnull().mean() * 100
    drop_high  = miss_rates[miss_rates > Config.HH_MISSING_RATE_DROP].index.tolist()
    if drop_high:
        print(f"  Dropping {len(drop_high)} Z_h columns with "
              f">{Config.HH_MISSING_RATE_DROP}% missing: {drop_high[:5]} ...")
        hh_for_join.drop(columns=drop_high, errors='ignore', inplace=True)

    # Add hh_ prefix to all feature columns; household_id stays unprefixed
    # so it can serve as the merge key.
    rename_map = {
        c: Config.HH_PREFIX + c
        for c in hh_for_join.columns if c != 'household_id'
    }
    hh_for_join.rename(columns=rename_map, inplace=True)
    hh_cols = [c for c in hh_for_join.columns if c != 'household_id']

    # Normalise merge key — strip trailing .0 so both sides match the
    # format produced by get_hid() (e.g. '123456' not '123456.0').
    hh_for_join['household_id'] = (
        hh_for_join['household_id'].astype(str).str.rstrip('.0'))
    df['_household_id_str'] = df['_household_id'].astype(str)

    # ── 6. Join Z_h to individual rows ───────────────────────────────────────
    # Diagnostic: check key overlap before merging
    hh_keys  = set(hh_for_join['household_id'])
    df_keys  = set(df['_household_id_str'].dropna())
    overlap  = len(hh_keys & df_keys)
    print(f"  Key overlap check: {overlap:,} of {len(df_keys):,} "
          f"person-household IDs matched in Z_h table")
    if overlap == 0:
        raise RuntimeError(
            "No household IDs matched — check .rstrip('.0') normalisation "
            "in both get_hid() and the batch loop.")
    print("  Joining Z_h to individual rows …")
    df = df.merge(
        hh_for_join.rename(columns={'household_id': '_household_id_str'}),
        on='_household_id_str',
        how='left',
    )
    df.drop(columns=['_household_id', '_household_id_str'], inplace=True)
    del hh_feat_df, hh_for_join; gc.collect()

    # ── 7. Summary ────────────────────────────────────────────────────────────
    th_dist = df['T_h'].value_counts(dropna=False).sort_index().to_dict()
    n_th1   = (df['T_h'] == 1).sum()
    n_th0   = (df['T_h'] == 0).sum()
    n_hh_na = df[hh_cols[0]].isna().sum() if hh_cols else 0

    print(f"\n  ── Step 1.5 summary ─────────────────────────────────────────")
    print(f"  T_h distribution          : {th_dist}")
    print(f"  T_h=1 members             : {n_th1:,}  "
          f"(pos rate: {(df.loc[df['T_h']==1,'label']==1).mean()*100:.1f}%)")
    print(f"  T_h=0 members             : {n_th0:,}  "
          f"(pos rate: {(df.loc[df['T_h']==0,'label']==1).mean()*100:.1f}% "
          f"— expected 0%)")
    print(f"  hh_ columns added         : {len(hh_cols)}")
    print(f"  Missing hh_ features      : {n_hh_na:,} rows")
    print(f"  ─────────────────────────────────────────────────────────────")

    return df

# STEP 2 – FEATURE SELECTION  (count features only, same as V2)

def chi_square_filter(df: pd.DataFrame,
                       candidate_cols: List[str],
                       target: pd.Series) -> List[str]:
    print(f"\n  Stage 1 – Chi-square filter on {len(candidate_cols):,} features …")
    kept = []
    n_batches = (len(candidate_cols) + Config.CHI2_BATCH_SIZE - 1) \
                // Config.CHI2_BATCH_SIZE

    for b in range(n_batches):
        cols  = candidate_cols[b*Config.CHI2_BATCH_SIZE:
                               (b+1)*Config.CHI2_BATCH_SIZE]
        batch = df[cols].copy().clip(lower=0).fillna(0)
        _, pv = chi2(batch, target)
        freq_ok = [
            (batch[c].value_counts(normalize=True).iloc[0]
             < Config.MAX_FREQ_THRESH
             if not batch[c].value_counts().empty else False)
            for c in cols
        ]
        kept.extend([
            c for c, p, ok in zip(cols, pv, freq_ok)
            if p < Config.CHI2_P_THRESHOLD and not np.isnan(p) and ok
        ])
        del batch; gc.collect()
        print(f"    Batch {b+1}/{n_batches}: {len(kept)}/{len(candidate_cols)} "
              f"kept so far")

    print(f"  -> After chi-square: {len(kept):,} features remain")
    return kept

def select_features(df: pd.DataFrame) -> Tuple[List[str], Dict]:
    print(f"\n{'='*70}")
    print("STEP 2 – FEATURE SELECTION  (count features only)")
    print(f"{'='*70}")

    count_cols       = [c for c in df.columns
                        if c.startswith(Config.COUNT_PREFIXES)]
    after_chi2       = chi_square_filter(df, count_cols, df['label'])
    after_epi, stats = apply_epidemiological_filter(after_chi2)

    print(f"\n  Summary:")
    for prefix, key in [('lmed_','lmed'),('ov_','ov'),
                         ('sv_','sv'),('contact_','contact')]:
        a2 = sum(1 for c in after_chi2 if c.startswith(prefix))
        print(f"    {prefix:<13} after chi2: {a2:>6}  "
              f"after epi: {stats[f'{key}_kept']}/{stats[f'{key}_total']}")
    print(f"  -> Final count features: {len(after_epi):,}")

    return after_epi, {
        'count_features_initial': len(count_cols),
        'after_chi2':             len(after_chi2),
        'after_epi_filter':       len(after_epi),
        'epi_stats':              stats,
        'selected_count_cols':    after_epi,
    }

# STEP 3 – ENCODE AND IMPUTE

def encode_core_features(
    df:       pd.DataFrame,
    mode:     str             = 'fit',
    encoders: Optional[Dict]  = None,
    medians:  Optional[Dict]  = None,
) -> Tuple[pd.DataFrame, Optional[Dict], Optional[Dict]]:
    """
    Encode individual core features (same as V2) PLUS household categorical
    and numeric imputation for all hh_ features.

    Individual encoding (unchanged from V2):
      FodelseArMan  -> age_2020
      Boendeform    -> OHE  (individual-level, encoder key: 'ind_boende_ohe')
      Fodelseland   -> frequency encoding
      UtlSvBakg     -> label encoding
      numeric       -> median imputation + log transform
      T_h           -> cast int, impute 0

    Household encoding (NEW, applied to hh_* columns):
      hh_ numeric        -> median imputation (key: 'hh_med_{col}')
      hh_ log cols       -> log1p for financial/area features
      Excluded from Z_h  : Boendeform_mode, Boarea_Person_hh,
                           AntalBarnUnder18_hh (duplicates of X_i)
    """
    if mode == 'fit':
        encoders = {}
        medians  = {}

    # ─── Individual: Age ────────────────────────────────────────────────────
    if 'FodelseArMan' in df.columns:
        birth = pd.to_numeric(df['FodelseArMan'], errors='coerce')
        df['age_2020'] = (2020 - (birth // 100)).astype(float)
        if mode == 'fit':
            medians['age_2020'] = df['age_2020'].median()
        df['age_2020'] = df['age_2020'].fillna(medians['age_2020'])
        df.drop(columns=['FodelseArMan'], inplace=True)

    # ─── Individual: Boendeform OHE ─────────────────────────────────────────
    if 'Boendeform' in df.columns:
        col = df['Boendeform'].fillna('unknown').astype(str)
        if mode == 'fit':
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            arr = ohe.fit_transform(col.values.reshape(-1, 1))
            encoders['ind_boende_ohe'] = ohe
        else:
            ohe = encoders['ind_boende_ohe']
            arr = ohe.transform(col.values.reshape(-1, 1))
        bcols = [f'Boendeform_{c}' for c in ohe.categories_[0]]
        df.drop(columns=['Boendeform'], inplace=True)
        df = pd.concat(
            [df, pd.DataFrame(arr, columns=bcols, index=df.index)], axis=1)

    # ─── Individual: Fodelseland frequency ──────────────────────────────────
    if 'Fodelseland' in df.columns:
        col = df['Fodelseland'].fillna('unknown').astype(str)
        if mode == 'fit':
            encoders['fodelse_freq'] = col.value_counts(normalize=True).to_dict()
        df['Fodelseland_freq'] = col.map(encoders['fodelse_freq']).fillna(0.0)
        df.drop(columns=['Fodelseland'], inplace=True)

    # ─── Individual: UtlSvBakg label encoding ───────────────────────────────
    if 'UtlSvBakg' in df.columns:
        col = df['UtlSvBakg'].fillna('unknown').astype(str)
        if mode == 'fit':
            le = LabelEncoder()
            df['UtlSvBakg_enc'] = le.fit_transform(col)
            encoders['utlsv_le'] = le
        else:
            le    = encoders['utlsv_le']
            known = set(le.classes_)
            df['UtlSvBakg_enc'] = col.apply(
                lambda x: int(le.transform([x])[0]) if x in known else -1)
        df.drop(columns=['UtlSvBakg'], inplace=True)

    # ─── Individual: numeric median imputation ───────────────────────────────
    for c in ['Kon', 'AntalBarnUnder18', 'Boarea_Person',
              'DispInk04', 'DispInkFam04', 'TRYGG_1', 'TRYGG_total']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            if mode == 'fit':
                medians[c] = df[c].median()
            df[c] = df[c].fillna(medians[c])

    # ─── Individual: T_h ────────────────────────────────────────────────────
    if 'T_h' in df.columns:
        df['T_h'] = pd.to_numeric(df['T_h'], errors='coerce').fillna(0).astype(int)

    # ─── Individual: log transforms ─────────────────────────────────────────
    for c in ['Boarea_Person', 'DispInk04', 'DispInkFam04']:
        if c in df.columns:
            df[f'{c}_log'] = np.log1p(df[c].clip(lower=0))
            df.drop(columns=[c], inplace=True)

    # ─── Household: numeric median imputation for all hh_ columns ───────────
    # hh_Boendeform_mode has been removed from the feature set — it
    # duplicates the individual-level Boendeform (same dwelling).
    hh_num_cols = [
        c for c in df.columns
        if c.startswith(Config.HH_PREFIX)
        and df[c].dtype in (np.float64, np.float32, float,
                             np.int64, np.int32, np.int8)
    ]
    for c in hh_num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        key = f'hh_med_{c}'
        if mode == 'fit':
            medians[key] = df[c].median()
        df[c] = df[c].fillna(medians.get(key, df[c].median()))

    # ─── Household: log transforms for financial / area columns ─────────────
    # Removed from log list: Boarea_Person_hh (duplicate of individual
    # Boarea_Person), index_mean_Boarea (removed non-aggregated feature).
    hh_log_bases = [
        'total_Boarea', 'crowding_index',
        'mean_DispInk04', 'mean_DispInkFam04',
        'index_mean_DispInk04',
    ]
    for base in hh_log_bases:
        col = Config.HH_PREFIX + base
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))

    return df, encoders, medians

# STEP 4 – STANDARDISE

def standardise(
    train: pd.DataFrame,
    val:   Optional[pd.DataFrame],
    test:  Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Fit StandardScaler on train, apply to val and test.

    Not scaled:
      Individual : encoded categorical, binary dummies, T_h, label, id, dates
      Household  : binary indicators (hh_is_*, hh_has_*, hh_any_*,
                   hh_three_*, hh_multi*, hh_gender_*, hh_all_*,
                   hh_Boendeform_*) and integer counts
    """
    skip_exact = {
        'label', 'person_id', 'T_h',
        f'{Config.HH_PREFIX}household_size',
        f'{Config.HH_PREFIX}index_cases_count',
        f'{Config.HH_PREFIX}male_count',
        f'{Config.HH_PREFIX}female_count',
        f'{Config.HH_PREFIX}age_0_17_count',
        f'{Config.HH_PREFIX}age_18_64_count',
        f'{Config.HH_PREFIX}age_65plus_count',
        f'{Config.HH_PREFIX}TRYGG_1_sum',
        f'{Config.HH_PREFIX}TRYGG_total_sum',
        f'{Config.HH_PREFIX}Fodelseland_diversity',
        # Removed: hh_AntalBarnUnder18_hh, hh_Boarea_Person_hh,
        # hh_Boendeform_* — these duplicated X_i individual features.
    }
    skip_suffixes = ('_enc', '_id', 'IndexDate')
    skip_prefixes = (
        'Boendeform_',
        f'{Config.HH_PREFIX}is_',
        f'{Config.HH_PREFIX}has_',
        f'{Config.HH_PREFIX}any_',
        f'{Config.HH_PREFIX}three_',
        f'{Config.HH_PREFIX}multi',
        f'{Config.HH_PREFIX}gender_',
        f'{Config.HH_PREFIX}all_',
        f'{Config.HH_PREFIX}index_has_',
        f'{Config.HH_PREFIX}index_any_',
    )

    cols_to_scale = [
        c for c in train.columns
        if c not in skip_exact
        and train[c].dtype in (np.float32, np.float64, float)
        and not any(c.endswith(s)   for s in skip_suffixes)
        and not any(c.startswith(s) for s in skip_prefixes)
    ]

    print(f"\n  Standardising {len(cols_to_scale)} continuous features …")
    ind_count = sum(1 for c in cols_to_scale
                    if not c.startswith(Config.HH_PREFIX))
    hh_count  = sum(1 for c in cols_to_scale
                    if c.startswith(Config.HH_PREFIX))
    print(f"    Individual: {ind_count}  |  Household (hh_): {hh_count}")

    for col in tqdm(cols_to_scale, desc="  Scaling"):
        scaler     = StandardScaler()
        train_vals = np.nan_to_num(train[[col]].values, nan=0.0,
                                    posinf=0.0, neginf=0.0)
        scaler.fit(train_vals)
        train[col + '_std'] = scaler.transform(train_vals).flatten()
        train.drop(columns=[col], inplace=True)
        for split_df in [val, test]:
            if split_df is not None and col in split_df.columns:
                vals = np.nan_to_num(split_df[[col]].values, nan=0.0,
                                      posinf=0.0, neginf=0.0)
                split_df[col + '_std'] = scaler.transform(vals).flatten()
                split_df.drop(columns=[col], inplace=True)

    gc.collect()
    return train, val, test

# MAIN PIPELINE

def run_framework2_pipeline() -> None:
    """
    Full Framework II dataset pipeline with individual (X_i) +
    household (Z_h) features.

    Steps
    -----
    1.   load_and_relabel()               ML labels (T_h=0 members kept)
    1b.  load_person_table_for_hh()       original labels for Z_h computation
    1.5  attach_household_features()      compute Z_h, annotate T_h, join
    2.   select_features()                chi-sq + epi filter on count cols
    3.   K-fold split
    4.   encode_core_features()           per-fold, fit on train
    5.   standardise()                    per-fold, fit on train
    6.   Save CSVs
    """
    print("=" * 70)
    print("PERSONAL-LEVEL DATASET CREATOR — FRAMEWORK II")
    print("Features: X_i (individual) + Z_h (household, hh_*) + T_h")
    print("Negative class: T_h=0 AND T_h=1 healthy members")
    print("=" * 70)

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    report_lines = []

    # ── Step 1: Load & relabel ─────────────────────────────────────────────
    df = load_and_relabel(
        Config.PERSON_TABLE_PATH, Config.ENCODING, Config.PERSON_CHUNK_SIZE)

    # ── Step 1b: Separate load for household feature computation ───────────
    print(f"\n{'='*70}")
    print("STEP 1b – LOADING PERSON TABLE (original labels, for Z_h)")
    print(f"{'='*70}")
    person_df_full = load_person_table_for_hh_features()

    # ── Load hushall_dict (shared by Step 1.5) ─────────────────────────────
    print("  Loading household membership pickle …")
    with open(Config.HUSHALL_PKL_PATH, 'rb') as f:
        hushall_dict = pickle.load(f)
    hushall_dict = {str(k).rstrip('.0'): v for k, v in hushall_dict.items()}
    print(f"  Membership entries: {len(hushall_dict):,}")

    # ── Step 1.5: Compute Z_h, annotate T_h, join ─────────────────────────
    df = attach_household_features(df, person_df_full, hushall_dict)
    force_cleanup(person_df_full, hushall_dict)

    # ── Step 2: Feature selection on count columns ─────────────────────────
    count_cols_selected, selection_report = select_features(df)

    hh_cols_in_df = [c for c in df.columns if c.startswith(Config.HH_PREFIX)]

    report_lines += [
        "=" * 60,
        "FEATURE SELECTION REPORT — Framework II (X_i + Z_h)",
        "=" * 60,
        "Individual X_i : core features + selected count features",
        "Household  Z_h : hh_* features (no chi-square, all retained)",
        f"T_h feature    : core (bypasses chi-square)",
        f"Total count features   : {selection_report['count_features_initial']:,}",
        f"After chi-square       : {selection_report['after_chi2']:,}",
        f"After epi filter       : {selection_report['after_epi_filter']:,}",
        f"hh_ columns            : {len(hh_cols_in_df)}",
    ]
    for k, v in selection_report['epi_stats'].items():
        report_lines.append(f"  {k}: {v}")
    report_lines.append("\nSelected count features:")
    for c in count_cols_selected:
        report_lines.append(f"  {c}")
    report_lines.append("\nHousehold features included (hh_ prefix):")
    for c in sorted(hh_cols_in_df):
        report_lines.append(f"  {c}")

    # ── Build working DataFrame ────────────────────────────────────────────
    work_cols = (
        ['person_id', 'IndexDate', 'label']
        + [c for c in Config.CORE_FEATURES if c in df.columns]
        + count_cols_selected
        + hh_cols_in_df
    )
    work_cols = list(dict.fromkeys(c for c in work_cols if c in df.columns))
    df        = df[work_cols].copy()
    gc.collect()

    print(f"\n  Working DataFrame : {len(df):,} rows x {len(df.columns)} cols")
    print(f"  Individual features (pre-encoding) : "
          f"{len(df.columns) - len(hh_cols_in_df) - 3}")
    print(f"  Household features (hh_)           : {len(hh_cols_in_df)}")

    # ── Step 3: K-fold ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("STEP 3 – K-FOLD DATASET CREATION")
    print(f"{'='*70}")

    train_full, test_df = train_test_split(
        df, test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE, stratify=df['label'])
    del df; gc.collect()

    print(f"  Train pool : {len(train_full):,}  "
          f"(T_h=1: {(train_full['T_h']==1).sum():,} | "
          f"T_h=0: {(train_full['T_h']==0).sum():,})")
    print(f"  Test set   : {len(test_df):,}")

    skf   = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True,
                             random_state=Config.RANDOM_STATE)
    folds = list(skf.split(train_full, train_full['label']))

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        fold_num   = fold_idx + 1
        print(f"\n{'='*70}\n  FOLD {fold_num}/{Config.N_FOLDS}\n{'='*70}")

        train_fold = train_full.iloc[tr_idx].copy()
        val_fold   = train_full.iloc[val_idx].copy()

        print(f"  Train: {len(train_fold):,} | Val: {len(val_fold):,} "
              f"| Test: {len(test_df):,}")

        # Encode (fit on train only)
        train_fold, encoders, medians = encode_core_features(
            train_fold, mode='fit')
        val_fold,   _, _ = encode_core_features(
            val_fold,   mode='transform', encoders=encoders, medians=medians)
        test_copy = test_df.copy()
        test_copy, _, _ = encode_core_features(
            test_copy,  mode='transform', encoders=encoders, medians=medians)

        # Fill count NaNs
        for split in [train_fold, val_fold, test_copy]:
            cnt = [c for c in count_cols_selected if c in split.columns]
            split[cnt] = split[cnt].fillna(0)

        # Standardise
        train_fold, val_fold, test_copy = standardise(
            train_fold, val_fold, test_copy)

        # Column alignment (OHE may see different categories in splits)
        all_cols = sorted(
            set(train_fold.columns) & set(val_fold.columns) &
            set(test_copy.columns))
        train_fold = train_fold[all_cols]
        val_fold   = val_fold[all_cols]
        test_copy  = test_copy[all_cols]

        # Sanity checks
        assert 'T_h' in all_cols,    "T_h missing from fold!"
        assert 'label' in all_cols,  "label missing from fold!"
        hh_final = sum(1 for c in all_cols if c.startswith(Config.HH_PREFIX))
        assert hh_final > 0, "No hh_ features in fold!"

        train_path = os.path.join(Config.OUTPUT_DIR, f'train_fold_{fold_num}.csv')
        val_path   = os.path.join(Config.OUTPUT_DIR, f'val_fold_{fold_num}.csv')
        test_path  = os.path.join(Config.OUTPUT_DIR, f'test_fold_{fold_num}.csv')

        train_fold.to_csv(train_path, index=False, encoding=Config.ENCODING)
        val_fold.to_csv(val_path,     index=False, encoding=Config.ENCODING)
        test_copy.to_csv(test_path,   index=False, encoding=Config.ENCODING)

        print(f"  Saved -> {train_path}")
        print(f"  Final columns: {len(all_cols)}  "
              f"(hh_ features: {hh_final})")

        force_cleanup(train_fold, val_fold, test_copy, encoders, medians)

    # Save feature selection report
    report_path = os.path.join(Config.OUTPUT_DIR, 'feature_selection_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\n  Report -> {report_path}")

    print("\n" + "=" * 70)
    print("FRAMEWORK II PIPELINE COMPLETE")
    print(f"Output: {Config.OUTPUT_DIR}/")
    print()
    print("Feature layers in output datasets:")
    print("  X_i  — individual (demographic, socioeconomic, ATC/ICD codes)")
    print("  Z_h  — household aggregates prefixed 'hh_' (shared per household)")
    print("  T_h  — household transmission indicator (binary)")
    print()
    print("REMINDER:")
    print("  * Model estimates P(Y_i=1 | X_i, Z_h)  [marginal, stand-alone]")
    print("  * Do NOT multiply by household model output.")
    print("=" * 70)

if __name__ == '__main__':
    run_framework2_pipeline()
