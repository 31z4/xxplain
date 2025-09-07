import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import sqlglot
from sqlglot import exp

def _extract_features_from_sql(features, sql):
    try:
        query_tree = sqlglot.parse_one(sql)
    except Exception:
        return

    features['n_tables'] = len(list(query_tree.find_all(exp.Table)))
    features['n_joins'] = len(list(query_tree.find_all(exp.Join)))
    features['has_subquery'] = 1.0 if query_tree.find(exp.Subquery) else 0.0
    features['has_cte'] = 1.0 if query_tree.find(exp.CTE) else 0.0
    features['has_groupby'] = 1.0 if query_tree.find(exp.Group) else 0.0
    features['has_distinct'] = 1.0 if 'DISTINCT' in sql.upper() else 0.0
    features['has_orderby'] = 1.0 if query_tree.find(exp.Order) else 0.0
    features['has_window_fn'] = 1.0 if query_tree.find(exp.Window) else 0.0
    expressions = query_tree.find_all(exp.Expression)
    features['complexity_depth'] = max((node.depth for node in expressions), default=0.0)
    features['selectivity_est'] = 0.0
    joins = list(query_tree.find_all(exp.Join))
    features['has_cartesian'] = 1.0 if any(join.args.get('on') is None for join in joins) else 0.0
    features['is_update_or_delete'] = 1.0 if isinstance(query_tree, (exp.Update, exp.Delete)) else 0.0

def _extract_from_pg_stat_statement(features, data):
    calls = data.get('calls', 0)
    mean_exec_time = data.get('mean_exec_time', 0)
    wal_records = data.get('wal_records', 0)
    wal_fpi = data.get('wal_fpi', 0)
    wal_bytes = data.get('wal_bytes', 0)
    stddev_exec_time = data.get('stddev_exec_time', 0)
    total_exec_time = data.get('total_exec_time', 0)
    rows = data.get('rows', 0)
    shared_blks_hit = data.get('shared_blks_hit', 0)
    shared_blks_read = data.get('shared_blks_read', 0)
    shared_blks_written = data.get('shared_blks_written', 0)
    temp_blks_read = data.get('temp_blks_read', 0)
    temp_blks_written = data.get('temp_blks_written', 0)
    stddev_plan_time = data.get('stddev_plan_time', 0)
    mean_plan_time = data.get('mean_plan_time', 0)
    blk_read_time = data.get('blk_read_time', 0)
    blk_write_time = data.get('blk_write_time', 0)
    temp_blk_read_time = data.get('temp_blk_read_time', 0)
    temp_blk_write_time = data.get('temp_blk_write_time', 0)
    jit_generation_time = data.get('jit_generation_time', 0)
    jit_inlining_time = data.get('jit_inlining_time', 0)
    jit_optimization_time = data.get('jit_optimization_time', 0)
    jit_emission_time = data.get('jit_emission_time', 0)

    jit_total_time = jit_generation_time + jit_inlining_time + jit_optimization_time + jit_emission_time
    total_temp_io = temp_blks_read + temp_blks_written
    total_io = shared_blks_read + shared_blks_written + total_temp_io
    total_blks = shared_blks_hit + shared_blks_read
    total_io_time = blk_read_time + blk_write_time + temp_blk_read_time + temp_blk_write_time

    features['calls'] = calls
    features['mean_exec_time'] = mean_exec_time
    features['wal_records'] = wal_records
    features['wal_fpi'] = wal_fpi
    features['wal_bytes'] = wal_bytes
    features['stddev_exec_time'] = stddev_exec_time
    features['exec_time_per_row'] = total_exec_time / rows if rows else 0
    features['rows_per_call'] = rows / calls if calls else 0
    features['cache_hit_ratio'] = shared_blks_hit / total_blks if total_blks else 0
    features['io_intensity'] = (shared_blks_read + shared_blks_written) / total_exec_time if total_exec_time else 0
    features['total_io_time'] = total_io_time
    features['io_time_ratio'] = total_io_time / total_exec_time if total_exec_time else 0
    features['temp_io_ratio'] = total_temp_io / total_io if total_io else 0
    features['temp_heaviness'] = total_temp_io / calls if calls else 0
    features['plan_stability'] = stddev_plan_time / mean_plan_time if mean_plan_time else 0
    features['exec_stability'] = stddev_exec_time / mean_exec_time if mean_exec_time else 0
    features['plan_exec_ratio'] = mean_plan_time / mean_exec_time if mean_exec_time else 0
    features['jit_heaviness'] = jit_total_time / total_exec_time if total_exec_time else 0
    features['jit_time_ratio'] = jit_total_time / total_exec_time if total_exec_time else 0
    features['is_slow_query'] = 1.0 if mean_exec_time > 1000 else 0.0
    features['is_frequently_called'] = 0.0
    features['temp_disk_usage'] = total_temp_io / calls if calls else 0

def _extract_tables_info(features, sql, pg_stat_user_tables, pg_indexes):
    try:
        query_tree = sqlglot.parse_one(sql)
    except Exception:
        return

    # Извлекаем все таблицы из SQL-запроса
    tables = []
    for table in query_tree.find_all(exp.Table):
        name = table.name
        schema = table.args.get('db')
        if schema:
            full_name = f"{schema}.{name}"
        else:
            full_name = name
        tables.append(full_name)

    # Инициализируем суммы
    total_n_live_tup = 0
    total_seq_scan = 0
    total_idx_scan = 0
    total_n_indexes = 0

    # Обрабатываем каждую таблицу
    for table_name in tables:
        # Статистика из pg_stat_user_tables
        if pg_stat_user_tables is not None:
            if '.' in table_name:
                schema, name = table_name.split('.', 1)
                matching_rows = pg_stat_user_tables[(pg_stat_user_tables['schemaname'] == schema) & (pg_stat_user_tables['relname'] == name)]
            else:
                matching_rows = pg_stat_user_tables[pg_stat_user_tables['relname'] == table_name]
            if not matching_rows.empty:
                stat = matching_rows.iloc[0]
                total_n_live_tup += stat.get('n_live_tup', 0)
                total_seq_scan += stat.get('seq_scan', 0)
                total_idx_scan += stat.get('idx_scan', 0)

        # Индексы из pg_indexes
        if pg_indexes is not None:
            if '.' in table_name:
                schema, name = table_name.split('.', 1)
                table_indexes = pg_indexes[(pg_indexes['schemaname'] == schema) & (pg_indexes['tablename'] == name)]
            else:
                table_indexes = pg_indexes[pg_indexes['tablename'] == table_name]
            total_n_indexes += len(table_indexes)

    # Вычисляем производные метрики
    has_index = 1.0 if total_n_indexes > 0 else 0.0
    index_usage_ratio = total_idx_scan / (total_idx_scan + total_seq_scan) if (total_idx_scan + total_seq_scan) > 0 else 0.0
    table_bloat_ratio = 0.0  # Не реализовано, так как требует отдельного запроса

    # Добавляем в features
    features['table_size_rows'] = total_n_live_tup
    features['has_index'] = has_index
    features['n_indexes'] = total_n_indexes
    features['n_seq_scans'] = total_seq_scan
    features['n_idx_scans'] = total_idx_scan
    features['index_usage_ratio'] = index_usage_ratio
    features['table_bloat_ratio'] = table_bloat_ratio


def extract_features(
    query: str = "",
    pg_stat_row: Optional[Dict[str, Any]] = None,
    pg_stat_user_tables: Optional[pd.DataFrame] = None,
    pg_indexes: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    if pg_stat_row is None:
        pg_stat_row = {}
    features = defaultdict(float)
    _extract_features_from_sql(features, query)
    if pg_stat_row:
        _extract_from_pg_stat_statement(features, pg_stat_row)
    if pg_stat_user_tables is not None and pg_indexes is not None:
        _extract_tables_info(features, query, pg_stat_user_tables, pg_indexes)

    return dict(features)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: features.py benchmarks/tpc-h/queries/q02.sql | jq")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        sql = f.read()
    print(json.dumps(extract_features(sql)))
