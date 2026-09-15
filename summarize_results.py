#!/usr/bin/env python3
"""Print the useful MIA metrics and optionally the identified dataset IDs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


ID_FIELDS = {
    'correct': 'correctly_identified_ids',
    'members': 'recovered_member_ids',
    'non-members': 'correctly_rejected_non_member_ids',
    'false-positives': 'false_positive_ids',
    'missed-members': 'missed_member_ids',
}


AGGREGATE_METRICS = (
    'auc', 'tpr_at_fpr_0.001', 'tpr_at_fpr_0.01', 'tpr_at_fpr_0.1',
    'attack_accuracy', 'model_accuracy', 'recovered_member_count',
    'model_train_accuracy', 'model_test_accuracy', 'accuracy_generalization_gap',
    'model_train_loss', 'model_test_loss', 'loss_generalization_gap',
    'attack_optimization_seconds', 'attack_evaluation_seconds',
    'target_model_queries_total', 'target_model_queries_mean_per_audit_sample',
    'audit_samples_per_evaluation_second',
    'recovered_members_per_1000_target_queries',
)


def load_rows(path: Path) -> list[dict]:
    if path.is_dir():
        path = path / 'results.json'
    with path.open(encoding='utf-8') as handle:
        records = json.load(handle)
    return [{'__params__': record.get('params', {}), **record['metrics']} for record in records]


def load_recursive(path: Path) -> list[dict]:
    paths = [path] if path.is_file() else sorted(path.rglob('results.json'))
    if not paths:
        raise FileNotFoundError(f'No results.json files found below {path}')
    rows = []
    for result_path in paths:
        rows.extend(load_rows(result_path))
    return rows


def t_critical_95(degrees_of_freedom: int) -> float:
    # Two-sided Student-t 97.5th percentiles; normal approximation after 30 df.
    values = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306,
              2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120,
              2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064,
              2.060, 2.056, 2.052, 2.048, 2.045, 2.042)
    return values[degrees_of_freedom - 1] if degrees_of_freedom <= 30 else 1.96


def uncertainty(values: list[float], bounds: tuple[float | None, float | None] = (None, None)) -> dict:
    n = len(values)
    mean = statistics.fmean(values)
    if n < 2:
        return {'n': n, 'mean': mean, 'std': 0.0,
                'ci95_low': mean, 'ci95_high': mean}
    std = statistics.stdev(values)
    margin = t_critical_95(n - 1) * std / math.sqrt(n)
    low, high = mean - margin, mean + margin
    if bounds[0] is not None:
        low = max(bounds[0], low)
    if bounds[1] is not None:
        high = min(bounds[1], high)
    return {'n': n, 'mean': mean, 'std': std,
            'ci95_low': low, 'ci95_high': high}


def aggregate_rows(rows: list[dict]) -> list[dict]:
    identity_fields = (
        'attacker_mode', 'defender_mode', 'defender_model_name', 'dataset_name',
        'defender_split', 'attacker_split', 'audit_in_percentage',
        'configured_audit_samples', 'configured_shadow_datasets',
    )
    groups = {}
    for row in rows:
        params = {field: row.get(field) for field in identity_fields if row.get(field) is not None}
        for field in ('attacker_configuration', 'defense_configuration',
                      'defender_training_configuration', 'attacker_training_configuration'):
            if row.get(field) is not None:
                params[field] = row[field]
        params.update(row.get('__params__', {}))
        key = json.dumps(params, sort_keys=True, default=str)
        groups.setdefault(key, {'params': params, 'rows': []})['rows'].append(row)

    aggregated = []
    for key, grouped in sorted(groups.items()):
        group = grouped['rows']
        output = {'group': grouped['params'], 'num_runs': len(group),
                  'seeds': sorted({row.get('seed') for row in group if row.get('seed') is not None}),
                  'metrics': {}}
        for metric in AGGREGATE_METRICS:
            values = [float(row[metric]) for row in group
                      if row.get(metric) is not None and isinstance(row.get(metric), (int, float))]
            if values:
                bounded_ratios = {'auc', 'tpr_at_fpr_0.001', 'tpr_at_fpr_0.01',
                                  'tpr_at_fpr_0.1', 'attack_accuracy', 'model_accuracy',
                                  'model_train_accuracy', 'model_test_accuracy'}
                if metric in bounded_ratios:
                    bounds = (0.0, 1.0)
                elif metric in {'accuracy_generalization_gap', 'loss_generalization_gap'}:
                    bounds = (None, None)
                else:
                    bounds = (0.0, None)
                output['metrics'][metric] = uncertainty(values, bounds=bounds)
        aggregated.append(output)
    return aggregated


def save_aggregates(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'aggregate_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(rows, handle, indent=2)
    flat_rows = []
    for row in rows:
        flat = {**row['group'], 'num_runs': row['num_runs'],
                'seeds': ','.join(str(seed) for seed in row['seeds'])}
        for metric, stats in row['metrics'].items():
            for statistic, result in stats.items():
                flat[f'{metric}_{statistic}'] = result
        flat_rows.append(flat)
    columns = sorted({key for row in flat_rows for key in row})
    with (output_dir / 'aggregate_summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat_rows)


def value(row: dict, name: str) -> str:
    item = row.get(name)
    if isinstance(item, float):
        return f'{item:.4f}'
    return '-' if item is None else str(item)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Summarize a mia_bench results.json file or its containing directory.')
    parser.add_argument('path', type=Path)
    parser.add_argument('--ids', choices=ID_FIELDS, help='Also print one category of dataset IDs.')
    parser.add_argument('--aggregate', action='store_true',
                        help='Recursively aggregate repeated seeds/models with 95% confidence intervals.')
    parser.add_argument('--output-dir', type=Path,
                        help='Directory for aggregate_summary.json/csv (defaults to PATH).')
    args = parser.parse_args()

    if args.aggregate:
        rows = aggregate_rows(load_recursive(args.path))
        output_dir = args.output_dir or (args.path if args.path.is_dir() else args.path.parent)
        save_aggregates(rows, output_dir)
        for row in rows:
            print(f"{row['group']} ({row['num_runs']} runs; seeds={row['seeds']})")
            for metric, stats in row['metrics'].items():
                print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                      f"(95% CI {stats['ci95_low']:.4f}–{stats['ci95_high']:.4f})")
        print(f'Aggregate files written to {output_dir}')
        return

    rows = load_rows(args.path)
    columns = ('auc', 'model_accuracy', 'attack_accuracy', 'balanced_accuracy',
               'precision', 'recall', 'f1', 'tpr_at_fpr_0.01',
               'true_positives', 'true_negatives', 'false_positives', 'false_negatives')
    for index, row in enumerate(rows, 1):
        params = row.get('__params__', {})
        print(f'Result {index}' + (f' ({params})' if params else ''))
        print(f"  AUC: {value(row, 'auc')} | model accuracy: {value(row, 'model_accuracy')} | "
              f"attack accuracy: {value(row, 'attack_accuracy')}")
        print(f"  balanced accuracy: {value(row, 'balanced_accuracy')} | "
              f"precision/recall/F1: {value(row, 'precision')}/{value(row, 'recall')}/{value(row, 'f1')}")
        print(f"  TP/TN/FP/FN: {value(row, 'true_positives')}/{value(row, 'true_negatives')}/"
              f"{value(row, 'false_positives')}/{value(row, 'false_negatives')} | "
              f"TPR@1%FPR: {value(row, 'tpr_at_fpr_0.01')}")
        print(f"  train/test accuracy: {value(row, 'model_train_accuracy')}/"
              f"{value(row, 'model_test_accuracy')} | generalization gap: "
              f"{value(row, 'accuracy_generalization_gap')}")
        print(f"  attack optimize/evaluate seconds: {value(row, 'attack_optimization_seconds')}/"
              f"{value(row, 'attack_evaluation_seconds')} | target queries: "
              f"{value(row, 'target_model_queries_total')}")
        if args.ids:
            field = ID_FIELDS[args.ids]
            print(f'  {field}: {row.get(field, [])}')


if __name__ == '__main__':
    main()
