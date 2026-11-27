"""
Stage 3: score the live base and write the ranked lead list.

Trains the shipped model on the balanced training frame, scores every active
subscriber, calibrates the probability back to the true prior (tau), ranks, and
keeps decile 1 as the lead list the campaign dials.

Calibration:  p_c = (tau * p) / ( tau * p + (1 - tau) * (1 - p) )
It is monotonic in p, so it rescales the score without changing the ranking.

Run:  python infer_mode.py
"""
import os
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
import config
from common import spark_session, feature_columns
import features as FE
from test_mode import train, read_out


def build_inference_frame(spark):
    """All active subscribers with the full 59-feature vector (no label)."""
    base = FE.build_base(spark)
    vas, pi, bal = FE.build_vas(spark), FE.build_prior_ins(spark), FE.build_balance(spark)
    return FE.stack_tiers(base, vas, pi, bal)


def main():
    spark = spark_session("infer-mode")
    print("=== INFER MODE : score live base, calibrate (tau = {}), rank ===".format(config.TAU))

    df_train = read_out(spark, config.TRAIN_OUT)
    feats = feature_columns(df_train)
    model = train("gbt", feats, "label", df_train)

    infer = build_inference_frame(spark)
    first = udf(lambda v: float(v[0]), FloatType())
    scored = model.transform(infer).withColumn("prob1", 1 - first("probability"))

    tau = float(config.TAU)
    scored = scored.withColumn(
        "likelihood",
        (F.lit(tau) * F.col("prob1")) /
        (F.lit(tau) * F.col("prob1") + F.lit(1 - tau) * (1 - F.col("prob1"))))

    ranked = scored.withColumn("decile", F.ntile(10).over(Window.orderBy(F.col("prob1").desc())))
    leads = (ranked.filter(F.col("decile") == 1)
             .select("mobile_number", "ledger_acc_code", "likelihood")
             .orderBy(F.col("likelihood").desc()))

    n = leads.count()
    from common import save
    save(leads, config.INFER_OUT)
    print("decile-1 leads written: {}  ->  {}".format(n, config.INFER_OUT))
    leads.show(5, truncate=False)
    spark.stop()


if __name__ == "__main__":
    main()
