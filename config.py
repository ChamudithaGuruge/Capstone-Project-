"""
Central configuration for the insurance-VAS propensity pipeline.

This is the public, sanitised build. Logical source names below double as the
parquet file stems the synthetic generator writes under DATA_DIR. In production
you swap `common.load` to read your own warehouse tables instead (see common.py),
and the names/windows/package codes here are the only things that change.
"""
import os

# ---- paths (public build reads/writes local parquet) ----
DATA_DIR = os.environ.get("DATA_DIR", "./data")   # synthetic source tables
OUT_DIR  = os.environ.get("OUT_DIR",  "./out")    # pipeline outputs

# ---- logical source names (also parquet stems under DATA_DIR) ----
PROFILE = "profile"     # subscriber demography + status
REVENUE = "revenue"     # daily revenue per stream
VAS     = "vas"         # value-added-service charges
BALANCE = "balance"     # prepaid balance readings
OUTCOME = "outcome"     # call-centre outcomes (gold labels)

# ---- observation windows, expressed as days-ago from "today" ----
# Feature window sits strictly before the label window so they never overlap.
LABEL_LO,  LABEL_HI  = 1,   30     # did the subscriber activate? (label)
FEAT_LO,   FEAT_HI   = 31,  390    # features are built here
PRIOR_LO,  PRIOR_HI  = 391, 750    # prior-insurance history (the year before)
FEATURE_WINDOW_DAYS  = FEAT_HI - FEAT_LO + 1

# ---- package codes (generic / sanitised) ----
TARGET_PKGS = ["PKG_A", "PKG_B", "PKG_C"]   # insurance packages = the target
INS_PKGS    = ["PKG_A", "PKG_B"]            # subset counted as prior insurance
LOW_BAL     = 2000                          # low-balance threshold

# ---- labelling / sampling / calibration ----
GOLD_FRAC = 0.20          # share of confirmed outcomes frozen as the gold ruler
NEG_MODE  = "random"      # random | confirmed_only | baseline
NEG_SEED  = 122
TAU       = 0.014         # true-prior constant for probability calibration
SEED      = 8464

# ---- model hyperparameters ----
RF_PARAMS  = dict(numTrees=200, maxDepth=16, subsamplingRate=0.7,
                  minInstancesPerNode=10, maxBins=400, seed=SEED)
GBT_PARAMS = dict(maxIter=150, maxDepth=6, stepSize=0.1, subsamplingRate=0.7,
                  maxBins=400, seed=SEED)

# ---- output table/stem names ----
TRAIN_OUT = "train_frame"          # 59 features + label (balanced)
GOLD_OUT  = "gold_frame"           # 59 features + gold_label (frozen)
INFER_OUT = "scored_top_decile"    # live base, calibrated, decile 1

# ---- identifier columns (never used as model features) ----
ID_COLS = ["mobile_number", "ledger_acc_code"]
