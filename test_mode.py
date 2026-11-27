"""
Stage 2 (evaluation): honest validation on the frozen gold ruler.

Trains RF or GBT on the balanced training frame and scores it on the gold ruler
(real call outcomes: 1 = converted, 0 = declined). Reports the only two numbers
that matter for a dialer: gold AUC and the per-decile conversion lift.

  MODEL = 'gbt' reproduces the shipped model.
  MODEL = 'rf'  reproduces the baseline ensemble on the same frame.

An AUC far above the base model here usually means a feature is leaking the
label; investigate before trusting it.

Run:  MODEL=gbt python test_mode.py
"""
import os
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import config
from common import spark_session, feature_columns

MODEL = os.environ.get("MODEL", "gbt")


def read_out(spark, name):
    """Read a frame written by datagen (lives under OUT_DIR)."""
    return spark.read.parquet(os.path.join(config.OUT_DIR, name + ".parquet"))


def train(model, features, label_col, data):
    asm = VectorAssembler(inputCols=features, outputCol="features_vec")
    if model == "gbt":
        clf = GBTClassifier(labelCol=label_col, featuresCol="features_vec", **config.GBT_PARAMS)
    else:
        clf = RandomForestClassifier(labelCol=label_col, featuresCol="features_vec", **config.RF_PARAMS)
    return Pipeline(stages=[asm, clf]).fit(data)


def main():
    spark = spark_session("test-mode")
    print("=== TEST MODE  model = {} ===".format(MODEL))
    df_train = read_out(spark, config.TRAIN_OUT)
    gold = read_out(spark, config.GOLD_OUT)
    feats = feature_columns(df_train)
    print("num features:", len(feats))

    model = train(MODEL, feats, "label", df_train)

    first = udf(lambda v: float(v[0]), FloatType())
    pred = model.transform(gold).withColumn("prob1", 1 - first("probability")).cache()

    auc = BinaryClassificationEvaluator(labelCol="gold_label", rawPredictionCol="rawPrediction",
                                        metricName="areaUnderROC").evaluate(pred)
    base_rate = gold.filter(F.col("gold_label") == 1).count() / float(gold.count())
    deciled = pred.withColumn("decile", F.ntile(10).over(Window.orderBy(F.col("prob1").desc())))

    print("-" * 60)
    print("GOLD AUC [{}] : {:.4f}".format(MODEL, auc))
    print("gold base convert-rate : {:.4f}  (max decile lift = {:.2f}x)".format(base_rate, 1.0 / base_rate))
    print("decile | n | converts | rate | lift")
    for r in (deciled.groupBy("decile")
              .agg(F.count("*").alias("n"), F.sum("gold_label").alias("c"))
              .orderBy("decile").collect()):
        rate = r["c"] / float(r["n"]) if r["n"] else 0.0
        lift = rate / base_rate if base_rate else 0.0
        print("  {:>2d}   | {:>5d} | {:>6d} | {:.4f} | {:.2f}x".format(r["decile"], r["n"], int(r["c"]), rate, lift))
    print("-" * 60)
    print("On internal data: GBT ~0.6735 / D1 1.55x , RF ~0.6715 . AUC > 0.70 => suspect leakage.")
    spark.stop()


if __name__ == "__main__":
    main()
