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
    features_list = []

    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
        for row in reader:
            sql = row['query']
            plan = json.loads(row['plan'])
            plan_analyze = json.loads(row['plan_analyze'])
            features = extract_features(sql, pg_stat_user_tables_df, pg_indexes_df, plan)
            features_list.append({
                'features': json.dumps(features),
                # target values
                'time': get_time_from_plan(plan_analyze),
                'size': get_size_from_plan(plan_analyze),
            })
    features_df = pd.DataFrame(features_list)
    features_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: enrich_dataset.py datasets/train_query_plans.csv > train_features.csv")
        sys.exit(0)
    if len(sys.argv) == 2:
        enrich(sys.argv[1])
    elif len(sys.argv) == 3:
        enrich(sys.argv[1], sys.argv[2])
