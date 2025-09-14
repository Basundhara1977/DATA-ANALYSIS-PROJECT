#!/usr/bin/env python3
"""
Big Data Analysis at Scale — Dask / PySpark (CLI)
Usage:
  python big_data_analysis.py --engine dask --input "data/**/*.parquet" --out outputs --sample 0.05
  python big_data_analysis.py --engine spark --input "s3a://bucket/path/*.parquet" --out outputs --sample 0.0
"""
import argparse, os, time, sys, importlib

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["auto", "dask", "spark"], default="auto")
    p.add_argument("--input", required=True, help="Path/glob to Parquet/CSV (local or cloud)")
    p.add_argument("--out", default="outputs")
    p.add_argument("--sample", type=float, default=0.05, help="0.0 to 1.0; set 0 for full data")
    p.add_argument("--partitions", type=int, default=8)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    engine = args.engine
    if engine == "auto":
        engine = "dask" if importlib.util.find_spec("dask") else ("spark" if importlib.util.find_spec("pyspark") else None)
    if engine is None:
        print("Error: Neither Dask nor PySpark found. Please install one.", file=sys.stderr)
        sys.exit(1)

    if engine == "dask":
        import dask.dataframe as dd
        from dask.distributed import Client, LocalCluster
        cluster = LocalCluster()
        client = Client(cluster)

        # Detect file type
        if any(ext in args.input for ext in [".parquet", ".pq"]):
            df = dd.read_parquet(args.input, engine="pyarrow")
        else:
            df = dd.read_csv(args.input, assume_missing=True, dtype={"PULocationID": "float64", "DOLocationID": "float64"})

        cols = [c for c in ["tpep_pickup_datetime","tpep_dropoff_datetime","passenger_count","trip_distance","fare_amount","total_amount","PULocationID","DOLocationID"] if c in df.columns]
        df = df[cols]
        if args.sample and args.sample > 0 and args.sample < 1:
            df = df.sample(frac=args.sample, random_state=42)

        df["tpep_pickup_datetime"] = dd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
        df["tpep_dropoff_datetime"] = dd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")
        df = df.assign(duration_min=(df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()/60.0)
        df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 100) & (df["duration_min"] > 1) & (df["duration_min"] < 240)]
        df = df.assign(hour=df["tpep_pickup_datetime"].dt.hour)

        q = df[["trip_distance","duration_min"]].quantile([0.5, 0.95]).compute()
        hourly = df.groupby("hour").size().compute().sort_index()
        top_pu = df.groupby("PULocationID").size().nlargest(10).compute()

        q.to_csv(os.path.join(args.out, "percentiles.csv"))
        hourly.to_csv(os.path.join(args.out, "hourly_volume.csv"))
        top_pu.to_csv(os.path.join(args.out, "top_pickups.csv"))

        # scalability demo
        df2 = df.repartition(npartitions=args.partitions).persist()
        t0 = time.time()
        _ = df2["fare_amount"].mean().compute()
        print(f"Dask demo with {args.partitions} partitions took {time.time()-t0:.2f}s")

        df.to_parquet(os.path.join(args.out, "curated_parquet"), engine="pyarrow", write_index=False)

    else:  # spark
        from pyspark.sql import SparkSession, functions as F
        spark = (SparkSession.builder.appName("BigDataAnalysisDemo").getOrCreate())
        if any(ext in args.input for ext in [".parquet", ".pq"]):
            df = spark.read.parquet(args.input)
        else:
            df = spark.read.option("header", True).csv(args.input, inferSchema=True)

        cols = [c for c in ["tpep_pickup_datetime","tpep_dropoff_datetime","passenger_count","trip_distance","fare_amount","total_amount","PULocationID","DOLocationID"] if c in df.columns]
        df = df.select(*cols)
        if args.sample and args.sample > 0 and args.sample < 1:
            df = df.sample(False, args.sample, seed=42)

        df = (df
              .withColumn("tpep_pickup_datetime", F.to_timestamp(F.col("tpep_pickup_datetime")))
              .withColumn("tpep_dropoff_datetime", F.to_timestamp(F.col("tpep_dropoff_datetime")))
             )
        df = df.withColumn("duration_min", (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime"))/60.0)
        df = df.filter((F.col("trip_distance") > 0) & (F.col("trip_distance") < 100) & (F.col("duration_min") > 1) & (F.col("duration_min") < 240))
        df = df.withColumn("hour", F.hour("tpep_pickup_datetime"))

        q_td = df.approxQuantile("trip_distance", [0.5,0.95], 0.01)
        q_du = df.approxQuantile("duration_min", [0.5,0.95], 0.01)
        hourly = df.groupBy("hour").count().orderBy("hour")
        top_pu = df.groupBy("PULocationID").count().orderBy(F.desc("count")).limit(10)

        import pandas as pd
        pd.DataFrame([{"column":"trip_distance","p50":q_td[0],"p95":q_td[1]},{"column":"duration_min","p50":q_du[0],"p95":q_du[1]}]).to_csv(os.path.join(args.out, "percentiles.csv"), index=False)
        hourly.toPandas().to_csv(os.path.join(args.out, "hourly_volume.csv"), index=False)
        top_pu.toPandas().to_csv(os.path.join(args.out, "top_pickups.csv"), index=False)

        df2 = df.repartition(args.partitions).persist()
        t0 = time.time()
        _ = df2.agg(F.avg("fare_amount").alias("avg_fare")).collect()
        print(f"Spark demo with {args.partitions} partitions took {time.time()-t0:.2f}s")

        df.write.mode("overwrite").parquet(os.path.join(args.out, "curated_parquet"))

if __name__ == "__main__":
    main()
