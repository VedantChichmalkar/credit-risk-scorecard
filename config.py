"""Central configuration for the credit-risk scorecard project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Raw Lending Club file (download from Kaggle, see README). Any of these names works.
RAW_CANDIDATES = [
    DATA_DIR / "accepted_2007_to_2018Q4.csv",
    DATA_DIR / "accepted_2007_to_2018Q4.csv.gz",
    DATA_DIR / "loan.csv",
    DATA_DIR / "sample_lending_club.csv",      # synthetic smoke-test file
]

PREPARED = DATA_DIR / "prepared.pkl"
TARGET = "default"

# Time-based split: train on vintages before this year, test on this year and later.
TEST_FROM_YEAR = 2016

# Only loans with a resolved outcome are used.
GOOD_STATUSES = ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"]
BAD_STATUSES = ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"]

# Columns known only AFTER origination -> leakage. Never use as features.
LEAKAGE_COLS = [
    "funded_amnt", "funded_amnt_inv", "out_prncp", "out_prncp_inv",
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low", "debt_settlement_flag",
    "debt_settlement_flag_date", "settlement_status", "settlement_date",
    "settlement_amount", "settlement_percentage", "settlement_term",
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
    "deferral_term", "hardship_amount", "hardship_start_date", "hardship_end_date",
    "payment_plan_start_date", "hardship_length", "hardship_dpd",
    "hardship_loan_status", "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    "pymnt_plan", "policy_code", "collections_12_mths_ex_med",
]
# Identifiers / free text -> not features.
ID_TEXT_COLS = ["id", "member_id", "url", "desc", "title", "emp_title", "zip_code"]

# Origination-time features we keep (if present in the file).
NUMERIC_FEATURES = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs",
    "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies",
    "emp_length_years", "credit_history_years", "loan_to_income",
]
CATEGORICAL_FEATURES = [
    "term", "grade", "sub_grade", "home_ownership", "verification_status",
    "purpose", "addr_state", "application_type", "initial_list_status",
]

# WoE / scorecard settings
N_BINS = 10            # quantile bins for numeric features
MIN_BIN_FRAC = 0.02    # merge categorical levels rarer than this into "OTHER"
IV_THRESHOLD = 0.02    # drop features with IV below this (standard "unpredictive" cutoff)
CORR_THRESHOLD = 0.90  # drop the lower-IV feature of any pair with |corr| above this
PDO = 20               # points to double the odds
BASE_SCORE = 600       # score at BASE_ODDS
BASE_ODDS = 50         # good:bad odds at BASE_SCORE
RANDOM_STATE = 42
