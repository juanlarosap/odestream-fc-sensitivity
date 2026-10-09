"""Explicit source closure and read-only verification of authoritative calibration."""
from pathlib import Path
import math
from experiments.odestream_fc_multidataset_v3 import config as v3
from experiments.odestream_fc_multidataset_v3.checkpointing import read_json, sha
from experiments.odestream_fc_multidataset_v3.provenance import environment, verify_datasets

# Only reused modules in the actual import closure; no runner, CLI, status or corruption modules.
V3_FILES = ('__init__.py', 'config.py', 'checkpointing.py', 'provenance.py', 'runtime.py',
            'datasets.py', 'engine.py', 'model_adapter.py', 'models.py', 'neural_ode.py',
            'fc.py', 'losses.py', 'r50.py', 'timing.py', 'calibration.py', 'reporting.py')
ENV_KEYS = ('python_version', 'pytorch', 'numpy', 'pandas', 'sklearn',
            'intra_threads', 'inter_threads', 'deterministic', 'device')


def source_manifest():
    here = Path(__file__).parent
    paths = list(here.glob('*.py')) + [v3.ROOT/'experiments/odestream_fc_multidataset_v3'/f for f in V3_FILES]
    return {str(p.relative_to(v3.ROOT)): sha(p) for p in sorted(paths)}


def check_environment(previous, current):
    for key in ENV_KEYS:
        if previous[key] != current[key]:
            raise ValueError('Numerical environment mismatch: ' + key)


def verify_calibration(out, dataset):
    """Never runs calibration or writes v3 artifacts. Pins original, not current inventory."""
    from experiments.odestream_fc_multidataset_v3.calibration import derive
    from experiments.odestream_fc_multidataset_v3.losses import finite
    import numpy as np

    out = Path(out).resolve()
    if dataset not in v3.DATASETS:
        raise ValueError('Unknown dataset')
    protocol = read_json(out/'protocol.json')
    if protocol != v3.protocol():
        raise ValueError('Calibration is not from the clean finalized v3 protocol')
    ph = v3.digest(protocol)
    sources = read_json(out/'sources.json')
    source_hash = v3.digest(sources)
    for name in V3_FILES:
        rel = 'experiments/odestream_fc_multidataset_v3/'+name
        if sources.get(rel) != sha(v3.ROOT/rel):
            raise ValueError('Reused v3 source differs from calibration source: '+rel)
    plan = v3.master_plan()
    if read_json(out/'plan.json') != plan:
        raise ValueError('Original v3 plan mismatch')
    lookup = {n['id']: n for n in plan['nodes']}
    files = {name: sha(out/name) for name in
             ('protocol.json', 'plan.json', 'sources.json', 'execution_environment.json')}

    def record(identifier, suffix):
        rel = identifier+suffix
        path = out/rel
        value = read_json(path)
        node = lookup[identifier]
        if (value.get('complete') is not True or value.get('node') != node
                or value.get('protocol_hash') != ph or value.get('source_hash') != source_hash
                or value.get('dataset_sha256') != v3.registry()[dataset]['sha256']):
            raise ValueError('Calibration artifact identity mismatch: '+rel)
        expected_parents = {}
        for parent_id in node['parents']:
            parent = lookup[parent_id]
            parent_rel = parent_id+('.json' if parent['kind'] in ('calibration', 'lambda') else '/state.pt')
            expected_parents[parent_rel] = sha(out/parent_rel)
            files[parent_rel] = expected_parents[parent_rel]
            parent_receipt_rel = parent_id+'.lineage.json'
            parent_receipt = read_json(out/parent_receipt_rel)
            if (parent_receipt.get('protocol_hash') != ph or parent_receipt.get('node_id') != parent_id
                    or parent_receipt.get('artifacts', {}).get(parent_rel) != expected_parents[parent_rel]):
                raise ValueError('Calibration parent lineage mismatch')
            files[parent_receipt_rel] = sha(out/parent_receipt_rel)
        receipt_rel = identifier+'.lineage.json'
        receipt = read_json(out/receipt_rel)
        checksum = sha(path)
        expected = dict(protocol_hash=ph, node_id=identifier, parents=expected_parents,
                        artifacts={rel: checksum})
        if value.get('parents') != expected_parents or receipt != expected:
            raise ValueError('Calibration lineage mismatch: '+rel)
        files[rel], files[receipt_rel] = checksum, sha(out/receipt_rel)
        return value

    reference = record(f'datasets/{dataset}/calibration/reference', '.json')
    coefficient = record(f'datasets/{dataset}/calibration/rho_1.0', '.json')
    fixed = dict(seed=0, steps=128, batch_size=1, beta=.99, eps=1e-12, lambda_zero=True,
                 exclusion_tolerance=0., target_rows=[24,152], input_row_union=[0,151],
                 all_128_baseline_commits_exact=True,
                 initialization='separate validation-selected preparation Control warmup',
                 gradient_observations='TRAIN only; no online observations')
    if any(reference.get(k) != v for k, v in fixed.items()) or len(reference['rows']) != 128:
        raise ValueError('Invalid v3 calibration reference contract')
    finite(reference, 'calibration reference')
    ratios = []
    for i, row in enumerate(reference['rows']):
        if (row['G_base'] < 0 or row['G_cons'] < 0 or row['lambda_consistency'] != 0
                or row['student_update_count'] != i+1 or row['teacher_update_count'] != i+1
                or row['unscaled_gradient_ratio'] != row['G_cons']/(row['G_base']+1e-12)):
            raise ValueError('Invalid calibration gradient row')
        if row['G_cons'] > 0:
            ratios.append(row['unscaled_gradient_ratio'])
    if (len(ratios) < 2 or reference['valid_ratio_count'] != len(ratios)
            or reference['exact_zero_consistency_gradient_steps'] != 128-len(ratios)
            or reference['median_unscaled_gradient_ratio'] != float(np.median(ratios))):
        raise ValueError('Invalid calibration median/exclusion policy')
    expected = derive(reference, 1.0)
    if any(coefficient.get(k) != value for k, value in expected.items()):
        raise ValueError('rho_1.0 coefficient derivation mismatch')
    if not math.isfinite(coefficient['lambda_consistency']) or coefficient['lambda_consistency'] <= 0:
        raise ValueError('Invalid FC coefficient')
    return dict(dataset=dataset, dataset_sha256=v3.registry()[dataset]['sha256'],
                rho_target=1.0, lambda_consistency=coefficient['lambda_consistency'],
                original_protocol_hash=ph, original_source_hash=source_hash,
                coefficient_sha256=files[f'datasets/{dataset}/calibration/rho_1.0.json'],
                artifacts=files, execution_environment=read_json(out/'execution_environment.json'))


def verify_references(out, datasets):
    """All failures reported together; no silent partial FC authorization."""
    records, errors = {}, []
    for dataset in datasets:
        try:
            records[dataset] = verify_calibration(out, dataset)
        except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
            errors.append(f'{dataset}: {exc}')
    if errors:
        raise ValueError('Required calibration verification failed:\n'+'\n'.join(errors))
    return records
