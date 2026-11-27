"""
Feature engineering: 59 features = 43 base + 3 bolt-on tiers (16).

  base  (43) : demography + prepaid revenue behaviour (totals, averages,
               recency, frequency, seasonality, categorical indices)
  Tier-1 (6) : VAS affinity, excluding the target insurance packages
  Tier-3 (4) : prior-insurance history, from the window before the feature window
  Tier-5 (6) : prepaid balance behaviour

All windows come from config and never overlap the label window.
"""
import pyspark.sql.functions as F
import config

STREAMS = ["data_rev", "bundle_rev", "voicesms_rev", "voicesms_bundle_rev"]
_REC_SENTINEL = config.FEAT_HI + 1          # "no activity" recency value


def _in(col, lo, hi):
    return (F.col(col) >= lo) & (F.col(col) <= hi)


def build_base(spark):
    """43 base features keyed by (mobile_number, ledger_acc_code)."""
    from common import load
    prof = load(spark, config.PROFILE)
    rev = (load(spark, config.REVENUE)
           .filter(_in("days_ago", config.FEAT_LO, config.FEAT_HI))
           .withColumn("total_rev", sum(F.col(s) for s in STREAMS)))

    w = float(config.FEATURE_WINDOW_DAYS)
    aggs = []
    # ---- per-stream: total, daily average, recency, active days (16) ----
    for s in STREAMS:
        aggs += [
            F.sum(s).alias(s + "_total"),
            (F.sum(s) / w).alias(s + "_daily_avg"),
            F.min(F.when(F.col(s) > 0, F.col("days_ago"))).alias(s + "_recency"),
            F.countDistinct(F.when(F.col(s) > 0, F.col("days_ago"))).alias(s + "_active_days"),
        ]
    # ---- total stream (4) ----
    aggs += [
        F.sum("total_rev").alias("total_rev_sum"),
        (F.sum("total_rev") / w).alias("total_rev_daily_avg"),
        F.min(F.when(F.col("total_rev") > 0, F.col("days_ago"))).alias("total_rev_recency"),
        F.countDistinct(F.when(F.col("total_rev") > 0, F.col("days_ago"))).alias("total_rev_active_days"),
    ]
    # ---- seasonality (6) ----
    aggs += [
        F.sum(F.when(_in("days_ago", 31, 120), F.col("total_rev"))).alias("q1_rev"),
        F.sum(F.when(_in("days_ago", 121, 210), F.col("total_rev"))).alias("q2_rev"),
        F.sum(F.when(_in("days_ago", 211, 300), F.col("total_rev"))).alias("q3_rev"),
        F.sum(F.when(_in("days_ago", 301, 390), F.col("total_rev"))).alias("q4_rev"),
        F.sum(F.when((F.col("days_ago") % 7) < 5, F.col("total_rev"))).alias("weekday_rev"),
        F.sum(F.when((F.col("days_ago") % 7) >= 5, F.col("total_rev"))).alias("weekend_rev"),
    ]
    # ---- behavioural extras (10) ----
    aggs += [
        F.max("total_rev").alias("max_daily_total"),
        F.stddev("total_rev").alias("spend_std"),
        F.countDistinct(F.when(F.col("total_rev") > 0, F.col("days_ago"))).alias("total_active_days"),
        F.min(F.when(F.col("total_rev") > 0, F.col("days_ago"))).alias("overall_recency"),
        F.sum(F.when(_in("days_ago", 211, 390), F.col("total_rev"))).alias("first_half_rev"),
        F.sum(F.when(_in("days_ago", 31, 210), F.col("total_rev"))).alias("second_half_rev"),
    ]
    base = rev.groupBy("mobile_number", "ledger_acc_code").agg(*aggs)

    # derived extras that need the aggregates above (4)
    base = (base
            .withColumn("nonzero_stream_count",
                        sum((F.col(s + "_total") > 0).cast("int") for s in STREAMS))
            .withColumn("rev_ratio", F.col("second_half_rev") / (F.col("first_half_rev") + F.lit(1.0)))
            .withColumn("weekend_share", F.col("weekend_rev") / (F.col("total_rev_sum") + F.lit(1.0)))
            .withColumn("low_spend_flag", (F.col("total_rev_sum") < F.lit(50.0)).cast("int")))

    # fill recency sentinels and nulls
    rec_cols = [s + "_recency" for s in STREAMS] + ["total_rev_recency", "overall_recency"]
    base = base.fillna(_REC_SENTINEL, subset=rec_cols)
    base = base.fillna(0.0)

    # ---- demography + categorical from profile (3 + 4) ----
    prof_sel = prof.select(
        "mobile_number", "ledger_acc_code",
        F.col("gender").cast("int").alias("gender"),
        F.col("age").cast("int").alias("age"),
        F.col("connected_days").cast("int").alias("connected_days"),
        F.col("district_idx").cast("int").alias("district_idx"),
        F.col("dsd_idx").cast("int").alias("dsd_idx"),
        F.col("device_type_idx").cast("int").alias("device_type_idx"),
        F.col("hardware_type_idx").cast("int").alias("hardware_type_idx"),
    )
    return base.join(prof_sel, ["mobile_number", "ledger_acc_code"], "inner")


def build_vas(spark):
    """Tier-1 VAS affinity (6), excluding the target insurance packages."""
    from common import load
    vas = (load(spark, config.VAS)
           .filter(_in("days_ago", config.FEAT_LO, config.FEAT_HI))
           .filter(~F.col("package").isin(config.TARGET_PKGS)))
    return (vas.groupBy("mobile_number").agg(
        F.countDistinct("package").alias("vas_distinct_pkgs"),
        F.count(F.lit(1)).alias("vas_txn_count"),
        F.sum("deducted_amt").alias("vas_total_deducted"),
        F.min("days_ago").alias("vas_recency_days"),
        F.max("days_ago").alias("vas_tenure_days"),
    ).withColumn("vas_active_flag", (F.col("vas_recency_days") <= 60).cast("int")))


def build_prior_ins(spark):
    """Tier-3 prior-insurance (4), from the window before the feature window."""
    from common import load
    pi = (load(spark, config.VAS)
          .filter(_in("days_ago", config.PRIOR_LO, config.PRIOR_HI))
          .filter(F.col("package").isin(config.INS_PKGS)))
    return (pi.groupBy("mobile_number").agg(
        F.count(F.lit(1)).alias("prior_ins_txns"),
        F.sum("deducted_amt").alias("prior_ins_total"),
        F.min("days_ago").alias("prior_ins_recency_days"),
    ).withColumn("prior_ins_ever", F.lit(1)))


def build_balance(spark):
    """Tier-5 prepaid balance behaviour (6)."""
    from common import load
    bal = (load(spark, config.BALANCE)
           .filter(_in("days_ago", config.FEAT_LO, config.FEAT_HI)))
    return (bal.groupBy("mobile_number").agg(
        F.avg("balance").alias("bal_avg"),
        F.min("balance").alias("bal_min"),
        F.max("balance").alias("bal_max"),
        F.stddev("balance").alias("bal_std"),
        F.count(F.lit(1)).alias("bal_readings"),
        F.avg((F.col("balance") < config.LOW_BAL).cast("double")).alias("bal_low_share"),
    ))


def stack_tiers(frame, vas, pi, bal):
    """Left-join the three tiers onto a base frame and zero-fill absentees."""
    out = (frame
           .join(vas, "mobile_number", "left")
           .join(pi,  "mobile_number", "left")
           .join(bal, "mobile_number", "left"))
    # recency defaults for subscribers with no VAS / no prior insurance
    out = out.fillna({"vas_recency_days": _REC_SENTINEL,
                      "vas_tenure_days": 0,
                      "prior_ins_recency_days": config.PRIOR_HI + 1})
    return out.fillna(0.0)
