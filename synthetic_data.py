"""
Schema-matched synthetic data generator.

Produces the five source tables (profile, revenue, vas, balance, outcome) as
parquet under DATA_DIR so the whole pipeline runs end to end with no company
data. A single latent trait per subscriber drives both the features and the
outcome, so the models learn a real (but synthetic) signal. The reported
production numbers come from internal data, not from this generator.

Run:  python synthetic_data.py           # uses SYN_N (default 30000)
"""
import os
import numpy as np
from pyspark.sql.types import (StructType, StructField, StringType,
                               IntegerType, DoubleType)
import config
from common import spark_session

N = int(os.environ.get("SYN_N", "30000"))
RNG = np.random.default_rng(config.SEED)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate():
    spark = spark_session("synthetic-data")
    os.makedirs(config.DATA_DIR, exist_ok=True)

    z = RNG.standard_normal(N)                       # latent propensity trait
    mob = np.array(["07{:08d}".format(i) for i in range(N)])
    led = np.array(["L{:08d}".format(i) for i in range(N)])

    # ---------------- profile ----------------
    gender = RNG.integers(0, 2, N)
    age = np.clip((30 + 12 * _sigmoid(z) + RNG.normal(0, 6, N)).astype(int), 18, 85)
    connected_days = RNG.integers(90, 3200, N)
    district_idx = RNG.integers(0, 25, N)
    dsd_idx = RNG.integers(0, 300, N)
    device_type_idx = RNG.integers(0, 6, N)
    hardware_type_idx = RNG.integers(0, 9, N)
    profile_rows = list(zip(mob.tolist(), led.tolist(), gender.tolist(), age.tolist(),
                            connected_days.tolist(), district_idx.tolist(), dsd_idx.tolist(),
                            device_type_idx.tolist(), hardware_type_idx.tolist(),
                            [1] * N, ["PRE"] * N))
    profile_schema = StructType([
        StructField("mobile_number", StringType()), StructField("ledger_acc_code", StringType()),
        StructField("gender", IntegerType()), StructField("age", IntegerType()),
        StructField("connected_days", IntegerType()), StructField("district_idx", IntegerType()),
        StructField("dsd_idx", IntegerType()), StructField("device_type_idx", IntegerType()),
        StructField("hardware_type_idx", IntegerType()), StructField("is_active", IntegerType()),
        StructField("connection_type", StringType())])

    # ---------------- revenue (R daily rows per subscriber) ----------------
    R = 14
    scale = np.exp(0.4 * z)[:, None]                 # (N,1) spend multiplier
    bases = {"data_rev": 40.0, "bundle_rev": 60.0, "voicesms_rev": 30.0, "voicesms_bundle_rev": 25.0}
    present = RNG.random((N, R)) < 0.6               # some days have no spend
    days = RNG.integers(config.FEAT_LO, config.FEAT_HI + 1, (N, R))
    stream_vals = {s: RNG.exponential(b * scale, (N, R)) * present for s, b in bases.items()}
    mob_rep = np.repeat(mob, R); led_rep = np.repeat(led, R)
    revenue_rows = list(zip(
        mob_rep.tolist(), led_rep.tolist(), days.reshape(-1).tolist(),
        stream_vals["data_rev"].reshape(-1).round(2).tolist(),
        stream_vals["bundle_rev"].reshape(-1).round(2).tolist(),
        stream_vals["voicesms_rev"].reshape(-1).round(2).tolist(),
        stream_vals["voicesms_bundle_rev"].reshape(-1).round(2).tolist()))
    revenue_schema = StructType([
        StructField("mobile_number", StringType()), StructField("ledger_acc_code", StringType()),
        StructField("days_ago", IntegerType()),
        StructField("data_rev", DoubleType()), StructField("bundle_rev", DoubleType()),
        StructField("voicesms_rev", DoubleType()), StructField("voicesms_bundle_rev", DoubleType())])

    # ---------------- balance (B readings per subscriber) ----------------
    B = 8
    bal = np.maximum(50.0, 2500.0 * (1 + 0.3 * z)[:, None] + RNG.normal(0, 800, (N, B)))
    bdays = RNG.integers(config.FEAT_LO, config.FEAT_HI + 1, (N, B))
    balance_rows = list(zip(np.repeat(mob, B).tolist(), bdays.reshape(-1).tolist(),
                            bal.reshape(-1).round(2).tolist()))
    balance_schema = StructType([
        StructField("mobile_number", StringType()), StructField("days_ago", IntegerType()),
        StructField("balance", DoubleType())])

    # ---------------- vas (non-target, prior-insurance, and target activations) ----------------
    vas_rows = []
    other_pkgs = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]
    # non-target VAS: up to T slots, present with prob rising in z
    T = 4
    p_vas = _sigmoid(0.3 * z - 0.2)[:, None]
    vas_present = RNG.random((N, T)) < p_vas
    for t in range(T):
        idx = np.where(vas_present[:, t])[0]
        if idx.size:
            pk = RNG.choice(other_pkgs, idx.size)
            amt = RNG.exponential(50.0, idx.size).round(2)
            da = RNG.integers(config.FEAT_LO, config.FEAT_HI + 1, idx.size)
            vas_rows += list(zip(mob[idx].tolist(), pk.tolist(), amt.tolist(), da.tolist()))
    # prior insurance: present with prob rising in z, in the prior window
    p_prior = _sigmoid(z - 1.0)
    for slot in range(2):
        idx = np.where(RNG.random(N) < p_prior)[0]
        if idx.size:
            pk = RNG.choice(config.INS_PKGS, idx.size)
            amt = RNG.exponential(80.0, idx.size).round(2)
            da = RNG.integers(config.PRIOR_LO, config.PRIOR_HI + 1, idx.size)
            vas_rows += list(zip(mob[idx].tolist(), pk.tolist(), amt.tolist(), da.tolist()))
    # target activation (the label), low base rate, in the label window
    p_act = _sigmoid(-4.4 + 1.4 * z)
    act_idx = np.where(RNG.random(N) < p_act)[0]
    if act_idx.size:
        pk = RNG.choice(config.TARGET_PKGS, act_idx.size)
        amt = RNG.exponential(60.0, act_idx.size).round(2)
        da = RNG.integers(config.LABEL_LO, config.LABEL_HI + 1, act_idx.size)
        vas_rows += list(zip(mob[act_idx].tolist(), pk.tolist(), amt.tolist(), da.tolist()))
    vas_schema = StructType([
        StructField("mobile_number", StringType()), StructField("package", StringType()),
        StructField("deducted_amt", DoubleType()), StructField("days_ago", IntegerType())])

    # ---------------- outcome (call-centre confirmed, ~8% called) ----------------
    called = RNG.random(N) < 0.08
    target = (RNG.random(N) < _sigmoid(-0.8 + 1.2 * z)).astype(int)
    oidx = np.where(called)[0]
    outcome_rows = list(zip(mob[oidx].tolist(), target[oidx].tolist()))
    outcome_schema = StructType([
        StructField("mobile_number", StringType()), StructField("target", IntegerType())])

    # ---------------- write ----------------
    def write(rows, schema, name):
        path = os.path.join(config.DATA_DIR, name + ".parquet")
        spark.createDataFrame(rows, schema).write.mode("overwrite").parquet(path)
        print("wrote {:>8} rows -> {}".format(len(rows), path))

    write(profile_rows, profile_schema, config.PROFILE)
    write(revenue_rows, revenue_schema, config.REVENUE)
    write(balance_rows, balance_schema, config.BALANCE)
    write(vas_rows, vas_schema, config.VAS)
    write(outcome_rows, outcome_schema, config.OUTCOME)
    print("synthetic base: N = {} subscribers, activation rate = {:.3f}".format(N, p_act.mean()))
    spark.stop()


if __name__ == "__main__":
    generate()
