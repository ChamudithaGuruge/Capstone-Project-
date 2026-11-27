"""
Stage 1: build the modelling frames.

  1. base features (43) for every active subscriber
  2. label  = did they activate a target insurance package in the label window?
  3. gold ruler = 20% of call-confirmed subscribers, frozen out of training
  4. negatives = confirmed decliners first, then random fill to a 1:1 balance
  5. stack the three tiers -> 59 features, and write train + gold frames

Run:  python datagen.py
"""
import pyspark.sql.functions as F
import config
from common import spark_session, load, save, feature_columns
import features as FE


def build_label(spark, base):
    """label = 1 if a target-package charge falls in the label window, else 0."""
    acts = (load(spark, config.VAS)
            .filter((F.col("days_ago") >= config.LABEL_LO) & (F.col("days_ago") <= config.LABEL_HI))
            .filter(F.col("package").isin(config.TARGET_PKGS))
            .select("mobile_number").distinct()
            .withColumn("_act", F.lit(1)))
    return (base.join(acts, "mobile_number", "left")
                .withColumn("label", F.when(F.col("_act") == 1, 1).otherwise(0))
                .drop("_act"))


def main():
    spark = spark_session("datagen")
    print("=== STAGE 1: base + label + gold freeze + negatives (mode = {}) ===".format(config.NEG_MODE))

    base = build_label(spark, FE.build_base(spark)).cache()
    print("base feature count:", len(feature_columns(base.drop("label"))))

    # gold ruler: all call-confirmed subscribers, 20% frozen by a stable hash
    outcome = load(spark, config.OUTCOME).select("mobile_number",
                                                 F.col("target").cast("int").alias("gold_label")).distinct()
    gold = (outcome
            .withColumn("hb", F.abs(F.hash("mobile_number")) % 100)
            .filter(F.col("hb") < int(config.GOLD_FRAC * 100)).drop("hb"))
    gold_mobiles = gold.select("mobile_number").distinct()
    gold_frame = base.join(gold, "mobile_number", "inner").drop("label")

    # training pool excludes the frozen gold subscribers (no leakage into the ruler)
    pool = base.join(gold_mobiles, "mobile_number", "left_anti")
    pos = pool.filter(F.col("label") == 1)
    neg_all = pool.filter(F.col("label") == 0)
    said_no = outcome.filter(F.col("gold_label") == 0).select("mobile_number").distinct()
    confirmed_neg = neg_all.join(said_no, "mobile_number", "inner")
    n_pos, n_conf = pos.count(), confirmed_neg.count()
    print("n_pos: {}  confirmed_neg: {}".format(n_pos, n_conf))

    if config.NEG_MODE == "confirmed_only":
        neg_final = confirmed_neg
        pos_final = pos.orderBy(F.rand(config.SEED)).limit(n_conf)
    elif config.NEG_MODE == "baseline":
        neg_final = neg_all.orderBy(F.rand(config.NEG_SEED)).limit(n_pos)
        pos_final = pos
    else:  # "random" (production): confirmed decliners + random fill to 1:1
        neg_unk = neg_all.join(said_no, "mobile_number", "left_anti")
        fill = max(n_pos - n_conf, 0)
        neg_final = confirmed_neg.unionByName(neg_unk.orderBy(F.rand(config.NEG_SEED)).limit(fill))
        pos_final = pos

    downsampled = pos_final.unionByName(neg_final)
    print("downsampled rows:", downsampled.count())

    # Stage 2: stack the 3 surviving tiers onto BOTH frames
    print("=== STAGE 2: VAS (+6) -> prior-insurance (+4) -> balance (+6) ===")
    vas, pi, bal = FE.build_vas(spark), FE.build_prior_ins(spark), FE.build_balance(spark)
    train_final = FE.stack_tiers(downsampled, vas, pi, bal)
    gold_final = FE.stack_tiers(gold_frame, vas, pi, bal)

    print("final feature count:", len(feature_columns(train_final)))
    save(train_final, config.TRAIN_OUT)
    save(gold_final, config.GOLD_OUT)
    print("wrote {} and {}".format(config.TRAIN_OUT, config.GOLD_OUT))
    spark.stop()


if __name__ == "__main__":
    main()
