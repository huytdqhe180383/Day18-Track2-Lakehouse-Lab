# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB4 — Medallion Pipeline (Bronze → Silver → Gold)
#
# **Use case:** LLM observability — exact schema from slide §6 medallion frame.
# Maps to deliverable bullet 4 (the Milestone-1 Lakehouse artifact).
#
# Pre-req: ran `python /workspace/scripts/generate_data.py` (writes Bronze).

# %%
import sys
sys.path.append("/workspace/scripts")
from spark_session import get_spark
from pyspark.sql import functions as F, types as T
from delta.tables import DeltaTable

spark = get_spark("nb4_medallion")

BRONZE = "s3a://bronze/llm_calls_raw"
SILVER = "s3a://silver/llm_calls"
GOLD   = "s3a://gold/llm_daily_metrics"

def describe_delta(path: str, name: str) -> None:
    print(f"\n[{name}] {path}")
    (spark.sql(f"DESCRIBE DETAIL delta.`{path}`")
        .select("format", "numFiles", "sizeInBytes")
        .show(truncate=False))

# %% [markdown]
# ## Bronze — verify raw is loaded

# %%
bronze = spark.read.format("delta").load(BRONZE)
print("Bronze rows:", bronze.count())
bronze.printSchema()
bronze.show(2, truncate=80)
describe_delta(BRONZE, "BRONZE")

# %% [markdown]
# ## Silver — parse, validate, dedup
#
# Rules: drop rows with malformed JSON, dedupe by `request_id`, project typed columns.

# %%
parsed_schema = T.StructType([
    T.StructField("model", T.StringType()),
    T.StructField("user_id", T.StringType()),
    T.StructField("usage", T.StructType([
        T.StructField("input", T.IntegerType()),
        T.StructField("output", T.IntegerType()),
    ])),
    T.StructField("latency_ms", T.IntegerType()),
    T.StructField("status", T.StringType()),
])

silver_df = (
    bronze
    .withColumn("p", F.from_json("raw_json", parsed_schema))
    .where(F.col("p").isNotNull())
    .select(
        "request_id",
        "ts",
        F.col("p.model").alias("model"),
        F.col("p.user_id").alias("user_id"),
        F.col("p.usage.input").alias("prompt_tokens"),
        F.col("p.usage.output").alias("completion_tokens"),
        F.col("p.latency_ms").alias("latency_ms"),
        F.col("p.status").alias("status"),
        F.to_date("ts").alias("date"),
    )
    .dropDuplicates(["request_id"])
)

(silver_df.write.format("delta").mode("overwrite")
    .partitionBy("date")
    .save(SILVER))

describe_delta(SILVER, "SILVER")

bronze_n = bronze.count()
silver_n = spark.read.format("delta").load(SILVER).count()
print(f"Silver rows: {silver_n:,}  (Bronze {bronze_n:,} → dedup dropped {bronze_n - silver_n:,})")
assert silver_n < bronze_n, (
    "Silver has the same row count as Bronze — dedup did not run. "
    "Did you regenerate Bronze with the updated generator (which seeds retries)?"
)

# %% [markdown]
# ## Gold — aggregate to (date, model) metrics

# %%
silver = spark.read.format("delta").load(SILVER)

# Cost model — illustrative USD/M-token (NOT canonical)
COST = {
    "claude-haiku-4-5":   (0.80, 4.00),
    "claude-sonnet-4-6":  (3.00, 15.00),
    "claude-opus-4-7":    (15.00, 75.00),
}
cost_in  = F.create_map(*[x for k, v in COST.items() for x in (F.lit(k), F.lit(v[0]))])
cost_out = F.create_map(*[x for k, v in COST.items() for x in (F.lit(k), F.lit(v[1]))])

gold_df = (silver
    .groupBy("date", "model")
    .agg(
        F.percentile_approx("latency_ms", 0.5).alias("p50_latency_ms"),
        F.percentile_approx("latency_ms", 0.95).alias("p95_latency_ms"),
        F.sum("prompt_tokens").alias("total_prompt_tokens"),
        F.sum("completion_tokens").alias("total_completion_tokens"),
        (F.sum(F.when(F.col("status") != "ok", 1).otherwise(0))
            / F.count("*")).alias("error_rate"),
    )
    .withColumn(
        "cost_usd",
        (F.col("total_prompt_tokens")    * cost_in[F.col("model")]  / F.lit(1_000_000)) +
        (F.col("total_completion_tokens")* cost_out[F.col("model")] / F.lit(1_000_000))
    )
)

(gold_df.write.format("delta").mode("overwrite")
    .partitionBy("date")
    .save(GOLD))

# Z-ORDER by model for fast filter-by-model dashboards
spark.sql(f"OPTIMIZE delta.`{GOLD}` ZORDER BY (model)")

describe_delta(GOLD, "GOLD")

# %% [markdown]
# ## Verify Gold

# %%
gold = spark.read.format("delta").load(GOLD)
gold.orderBy("date", "model").show(20, truncate=False)

# Gold validation for rubric
date_count = gold.select("date").distinct().count()
model_count = gold.select("model").distinct().count()
row_count = gold.count()
expected = date_count * model_count
print(
    f"Gold rows: {row_count}  (dates={date_count}, models={model_count}, expected={expected})"
)
print("Date span OK (≥ 7):", date_count >= 7)
print("Model count OK (≥ 3):", model_count >= 3)
assert date_count >= 7, f"Gold has only {date_count} dates; rubric requires >= 7."
assert model_count >= 3, f"Gold has only {model_count} models; expected >= 3."
assert row_count == expected, (
    f"Gold row shape mismatch: rows={row_count}, expected dates*models={expected}."
)

nulls = gold.select(
    F.sum(F.col("p50_latency_ms").isNull().cast("int")).alias("p50_nulls"),
    F.sum(F.col("p95_latency_ms").isNull().cast("int")).alias("p95_nulls"),
    F.sum(F.col("cost_usd").isNull().cast("int")).alias("cost_nulls"),
    F.sum(F.col("error_rate").isNull().cast("int")).alias("error_nulls"),
).collect()[0]
print(
    "Nulls - p50:", nulls["p50_nulls"],
    "p95:", nulls["p95_nulls"],
    "cost:", nulls["cost_nulls"],
    "error:", nulls["error_nulls"],
)
print("Cost column populated:", gold.where("cost_usd > 0").count() > 0)
assert nulls["p50_nulls"] == 0 and nulls["p95_nulls"] == 0, "Latency aggregates contain NULL values."
assert nulls["cost_nulls"] == 0 and nulls["error_nulls"] == 0, "cost_usd/error_rate contains NULL values."
assert gold.where("cost_usd > 0").count() > 0, "cost_usd is not populated with positive values."

# %% [markdown]
# ## ✅ Deliverable check
# - [ ] All three tables exist in MinIO (`bronze/`, `silver/`, `gold/`)
# - [ ] Silver has fewer rows than Bronze (dedup worked)
# - [ ] Gold rows = (#dates × #models)
# - [ ] Cost column populated (non-zero)

# %%
spark.stop()
