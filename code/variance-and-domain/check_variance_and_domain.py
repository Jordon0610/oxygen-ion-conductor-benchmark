"""Recalculate the cohort variance components and the internal applicability-domain checks."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from kernels import chemistry, compress, fit, fit_geometry, normalized, reference, rho, select


def check(table, expected_file, keys, rtol=1e-7, atol=1e-8):
    expected = pd.read_csv(Path(__file__).parent / 'expected' / expected_file).set_index(keys).sort_index()
    actual = table.set_index(keys).sort_index()
    pd.testing.assert_index_equal(actual.index, expected.index)
    for column in expected:
        # Floating-point differences among near-tied distances can change a few ranks.
        tolerance = 2e-5 if column == 'rho' else atol
        np.testing.assert_allclose(actual[column], expected[column], rtol=rtol, atol=tolerance)


def variance_results(data, cohort, names):
    features = pd.read_csv(data / 'benchmark_features_312.csv', index_col='row_id', float_precision='round_trip')
    scores, _ = compress(features.loc[cohort.index, names].copy())
    frame = cohort.assign(normalized_doi=cohort.DOI.str.strip().str.lower())
    for index in range(scores.shape[1]):
        frame[f'PC{index + 1:02d}'] = scores[:, index]
    rows = []
    for model in ['M0', 'M1', 'M2', 'M3', 'M4']:
        result = fit(frame, model)
        if not result['converged']:
            raise ValueError(f'{model} failed to converge')
        for key in ['source_VPC', 'family_VPC', 'residual_VPC']:
            rows.append({'model': model, 'statistic': key, 'estimate': result[key]})
    return pd.DataFrame(rows)


def domain_results(data, cohort, features, names):
    splits = json.loads((data / 'benchmark_splits.json').read_text())
    compositions = {i: set(chemistry(formula)[0]) for i, formula in cohort.normalized_formula.items()}
    distance_frames = []
    thresholds = []
    for split in splits:
        protocol = split['protocol']
        if protocol not in ('P3', 'P4'):
            continue
        train, test = split['train_ids'], split['test_ids']
        if not set(train).isdisjoint(test):
            raise ValueError(f"Shared training/test rows in {split['split_id']}")
        geometry, coordinates = fit_geometry(features.loc[train, names], features.loc[test, names])
        train_groups = cohort.loc[train, 'connected_group_id'].to_numpy()
        train_elements = set().union(*(compositions[i] for i in train))
        unseen = np.array([not compositions[i].issubset(train_elements) for i in test])
        spaces = [('PCA', 'pca', 'train_PCs'), ('raw', 'raw', 'train_scaled')]
        for space, key, coords in spaces:
            if protocol == 'P4' and space == 'raw':
                continue
            group_ref = reference(coordinates[coords], train_groups)
            row_q = geometry[key]['q95']
            group_q = group_ref['qgroup']
            thresholds.append(dict(split_id=split['split_id'], space=space, q95_row=row_q, q95_group=group_q))
            distance = geometry[key]['d']
            for ref, q in [('row', row_q), ('group', group_q)]:
                ratio = normalized(distance, q)
                distance_frames.append(pd.DataFrame({
                    'split_id': split['split_id'], 'protocol': protocol, 'row_id': test, 'space': space,
                    'reference': ref, 'd_test': distance, 'normalized_distance': ratio,
                    'primary_in_domain': (ratio <= 1) & ~unseen,
                }))
    return pd.DataFrame(thresholds), pd.concat(distance_frames, ignore_index=True)


def summarize_p3(data, distances):
    predictions = pd.read_csv(data / 'held_out_predictions.csv')
    errors = predictions.loc[(predictions.protocol == 'P3') & (predictions.model == 'ExtraTreesRegressor'), ['split_id', 'row_id', 'ae']]
    records = []
    for (space, ref), instances in distances[distances.protocol == 'P3'].groupby(['space', 'reference']):
        joined = instances.merge(errors, on=['split_id', 'row_id'], how='left', validate='one_to_one').sort_values(['split_id', 'row_id'])
        if joined.ae.isna().any():
            raise ValueError('Missing held-out errors')
        ratio = joined.normalized_distance.to_numpy()
        ae = joined.ae.to_numpy()
        mask = joined.primary_in_domain.to_numpy()
        retained = select(np.arange(len(ae)), ratio, .8)
        records.append(dict(space=space, reference=ref, in_instances=mask.sum(), out_instances=(~mask).sum(),
                            in_MAE=ae[mask].mean(), out_MAE=ae[~mask].mean(), rho=rho(ratio, ae), MAE_80=ae[retained].mean()))
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd(), help='Directory containing primary-benchmark/')
    parser.add_argument('--output', type=Path, help='Optional directory for the recalculated CSV tables')
    args = parser.parse_args()
    data = args.repo_root / 'primary-benchmark'
    cohort = pd.read_csv(data / 'experimental_cohort_312.csv', index_col='row_id')
    features = pd.read_csv(data / 'benchmark_features_312.csv', index_col='row_id')
    names = [name for name in features if not name.startswith('component_')]
    if len(names) != 139 or not cohort.index.is_unique or set(cohort.index) != set(features.index):
        raise ValueError('Expected matching row IDs and 139 chemical composition descriptors')
    with threadpool_limits(limits=1):
        variance = variance_results(data, cohort, names)
        thresholds, distances = domain_results(data, cohort, features, names)
        sensitivity = summarize_p3(data, distances)
    check(variance, 'variance.csv', ['model', 'statistic'])
    check(thresholds, 'thresholds.csv', ['split_id', 'space'])
    check(sensitivity, 'p3_sensitivity.csv', ['space', 'reference'], atol=1e-7)
    print(f'Verified {len(variance)} REML variance fractions, {len(thresholds)} pairs of training thresholds, '
          f'and {len(sensitivity)} P3 error summaries.')
    print(sensitivity.to_string(index=False))
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, table in [('variance', variance), ('thresholds', thresholds), ('distances', distances), ('p3_sensitivity', sensitivity)]:
            table.to_csv(args.output / f'{name}.csv', index=False)


if __name__ == '__main__':
    main()
