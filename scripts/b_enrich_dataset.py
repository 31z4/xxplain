import os
from pathlib import Path
import sys
import csv
import json
from dotenv import load_dotenv
import pandas as pd
import psycopg

# Specify the path to your custom .env file
dotenv_path = Path(".env.postgres")
load_dotenv(dotenv_path=dotenv_path)


# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml.features import extract_features, get_size_from_plan, get_time_from_plan

def get_conn(dsn: str):
    return psycopg.connect(dsn)


def enrich(input_file, output_file=sys.stdout):
    DSN = os.environ.get(
        "PG_DSN",
        f"postgresql://postgres:{os.getenv('POSTGRES_PASSWORD')}@localhost:5432/postgres",
    )

    conn = get_conn(DSN)
    pg_indexes_df = pd.read_sql('SELECT * FROM pg_indexes', conn)
    pg_stat_user_tables_df = pd.read_sql('SELECT * FROM pg_stat_user_tables', conn)

    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
        fieldnames = list(reader.fieldnames or []) + ['features', 'time', 'size']

        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
        writer.writeheader()

        for row in reader:
            plan = json.loads(row['plan'])
            plan_analyze = json.loads(row['plan_analyze'])
            features = extract_features(row['query'], pg_stat_user_tables_df, pg_indexes_df, plan, plan_analyze)
            row['features'] = json.dumps(features)
            # target values
            plan_analyze = json.loads(row['plan_analyze'])
            row['time'] = get_time_from_plan(plan_analyze)
            row['size'] = get_size_from_plan(plan_analyze)
            writer.writerow(row)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: enrich_dataset.py datasets/train_query_plans.csv > train_dataset.csv")
        sys.exit(0)
    enrich(sys.argv[1])
