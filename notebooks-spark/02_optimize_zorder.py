# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB2 — Small-File Problem & OPTIMIZE + ZORDER
#
# **Mục tiêu:** prove the 3–10× speedup claim from slide §5.
# Maps to deliverable bullet 2.

# %%
import sys, time, random
sys.path.append("/workspace/scripts")
from spark_session import get_spark
from delta.tables import DeltaTable

spark = get_spark("nb2_optimize_zorder")
path = "s3a://lakehouse/events_smallfiles"

# %% [markdown]
# ## 0. Reset path (idempotent re-run)
#
# Each run starts fresh — otherwise repeated appends keep growing the table
# and the benchmark drifts.

# %%
spark.sql(f"DROP TABLE IF EXISTS delta.`{path}`")
# Best-effort: the DROP above unregisters the catalog entry, but Delta files
# may persist in MinIO. Overwrite below resets the data.

# %% [markdown]
# ## 1. Manufacture the small-file problem
#
# Append 200 tiny batches → 200 small files. Realistic streaming-ingestion shape.

# %%
for batch in range(200):
    rows = [(i, random.choice(["click", "view", "scroll", "purchase"]),
             random.randint(1, 10000))
            for i in range(batch * 500, (batch + 1) * 500)]
    df = spark.createDataFrame(rows, ["event_id", "kind", "user_id"])
    mode = "overwrite" if batch == 0 else "append"
    df.write.format("delta").mode(mode).save(path)

def get_num_files(table_path: str) -> int:
    return (
        spark.sql(f"DESCRIBE DETAIL delta.`{table_path}`")
        .select("numFiles")
        .collect()[0][0]
    )

num_files_before = get_num_files(path)
print(f"numFiles BEFORE OPTIMIZE: {num_files_before}  (target ≥ 100)")
assert num_files_before >= 100, "Small-file problem not reproduced (need ≥ 100 files)."

# %% [markdown]
# ## 2. Benchmark BEFORE optimize

# %%
def bench(label):
    # Warm-up read so we measure query, not cold metadata fetch
    spark.read.format("delta").load(path).limit(1).count()
    df = (spark.read.format("delta").load(path)
            .where("user_id = 4242 AND kind = 'purchase'"))
    t0 = time.time()
    n = df.count()
    dt = time.time() - t0
    files_read = len(df.inputFiles())
    print(f"{label:25s}  count={n}  time={dt:.2f}s  files_read={files_read}")
    return dt, files_read

before, files_read_before = bench("BEFORE OPTIMIZE+ZORDER")

# %% [markdown]
# ## 3. OPTIMIZE + ZORDER

# %%
spark.sql(f"OPTIMIZE delta.`{path}` ZORDER BY (user_id)")

# %% [markdown]
# ## 4. Benchmark AFTER

# %%
after, files_read_after = bench("AFTER OPTIMIZE+ZORDER")
speedup = before / max(after, 1e-6)
prune_ratio = files_read_before / max(files_read_after, 1)
print(f"\nSpeedup: {speedup:.1f}×  (target ≥ 3×)")
print(f"Files-pruned ratio: {prune_ratio:.1f}×  (target ≥ 10×)")
meets_perf = (speedup >= 3.0) or (prune_ratio >= 10.0)
print("Perf criterion met:", meets_perf)
assert meets_perf, (
    f"Need speedup >= 3x OR files-pruned ratio >= 10x; got "
    f"speedup={speedup:.2f}x, pruned_ratio={prune_ratio:.2f}x"
)

# %% [markdown]
# ## 5. Inspect file count change

# %%
num_files_after = get_num_files(path)
print(f"numFiles AFTER OPTIMIZE:  {num_files_after}")
print(f"Files reduced: {num_files_before - num_files_after}")
spark.sql(f"DESCRIBE DETAIL delta.`{path}`").select(
    "numFiles", "sizeInBytes"
).show()
assert num_files_after < num_files_before, (
    f"Expected fewer files after OPTIMIZE; before={num_files_before}, after={num_files_after}."
)

# %% [markdown]
# ## ✅ Deliverable check
# - [ ] Speedup ≥ 3×
# - [ ] `numFiles` dropped substantially after OPTIMIZE
# - [ ] Screenshot the printed comparison

# %%
spark.stop()
