import os
import json
import sys
import re
from typing import List
import pandas as pd
import psycopg
import csv
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from ml.features import extract_features

# Specify the path to your custom .env file
dotenv_path = Path(".env.postgres")
load_dotenv(dotenv_path=dotenv_path)


def get_conn(dsn: str):
    return psycopg.connect(dsn)


def get_queries_with_stat(conn):
    # Execute following SQL
    #     SELECT *
    #     FROM pg_stat_statements
    #     WHERE query LIKE 'SELECT%';

    # then save results as list of dicts and return it
    sql = """
    SELECT queryid,
      mean_exec_time as time,
      (shared_blks_hit + shared_blks_read) * current_setting('block_size')::int AS size,
      query
    FROM pg_stat_statements
    WHERE query LIKE 'SELECT%';
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        data = [row for row in cur.fetchall()]
    return data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: pg_stat_statements_to_dataset.py train_dataset.csv")
        sys.exit(0)

    DSN = os.environ.get(
        "PG_DSN",
        f"postgresql://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/postgres",
    )

    conn = get_conn(DSN)
    pg_stat_statements_df = pd.read_sql("SELECT * FROM pg_stat_statements WHERE query NOT LIKE 'EXPLAIN%'", conn)
    pg_indexes_df = pd.read_sql('SELECT * FROM pg_indexes', conn)
    pg_stat_user_tables_df = pd.read_sql('SELECT * FROM pg_stat_user_tables', conn)

    # Process each row in pg_stat_statements_df
    features_list = []
    for idx, row in tqdm(pg_stat_statements_df.iterrows(), total=len(pg_stat_statements_df)):
        row_dict = row.to_dict()
        features = extract_features(row['query'], row_dict, pg_stat_user_tables_df, pg_indexes_df)
        features_list.append(features)

    # Create dataframe from features and save to CSV
    features_df = pd.DataFrame(features_list)
    features_df.to_csv(sys.argv[1], index=False)

    conn.close()
