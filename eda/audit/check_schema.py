
import duckdb
con = duckdb.connect()
for f in ["user_daily_stats", "item_daily_stats", "user_item_daily",
          "user_category_affinity", "user_city_affinity"]:
    print(f"--- {f} ---")
    df = con.execute(
        f"SELECT * FROM read_parquet('D:/Datathon_Model/agg/{f}.parquet') LIMIT 1"
    ).fetchdf()
    print("columns:", list(df.columns))
    print(df.dtypes.to_dict())
    print()

