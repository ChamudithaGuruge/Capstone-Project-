"""Shared helpers: Spark session and table IO.

Public build reads/writes parquet under config.DATA_DIR / config.OUT_DIR.
To run against a real warehouse, replace the body of `load` with
`return spark.table(name)` and `save` with `df.write.saveAsTable(name)`.
"""
import os
import config


def spark_session(app="insurance-vas-propensity"):
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder
             .appName(app)
             .config("spark.sql.shuffle.partitions", "16")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load(spark, name):
    """Load a logical source table. Public build: synthetic parquet."""
    path = os.path.join(config.DATA_DIR, name + ".parquet")
    return spark.read.parquet(path)


def save(df, name, out_dir=None):
    """Persist a pipeline output. Returns the path written."""
    out_dir = out_dir or config.OUT_DIR
    path = os.path.join(out_dir, name + ".parquet")
    df.write.mode("overwrite").parquet(path)
    return path


def feature_columns(df):
    """Model features = every column that is not an id, a label or a helper."""
    drop = set(config.ID_COLS) | {"label", "gold_label", "hash_bucket"}
    return [c for c in df.columns if c not in drop]
