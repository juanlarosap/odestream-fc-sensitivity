"""Corruption reports, read-only status, and data-only validation (no training)."""
import csv
import io
from pathlib import Path

from . import config as v3
from . import corruption30 as c
from .checkpointing import atomic, read_json, sha, write_json
from .reporting import summary


def status(out):
    from .status import snapshot
    state = snapshot(out)  # Reuse read-only local/remote process and stale-state checks.
    if not (Path(out) / 'state.json').exists():
        state.update(experiment_id=c.EXPERIMENT_ID, protocol_hash=v3.digest(c.protocol()))
    if state['protocol_hash'] != v3.digest(c.protocol()) or state['experiment_id'] != c.EXPERIMENT_ID:
        raise ValueError('Status protocol mismatch')
    lines = [c.EXPERIMENT_ID, 'Status: ' + state['status'], 'Stage: ' + state.get('current_stage', 'NOT_STARTED')]
    for title, key in [('Scientific warm-ups', 'completed_warmups'), ('Online trajectories', 'completed_online_trajectories')]:
        n = state.get(key, 0)
        if not 0 <= n <= 30:
            raise ValueError('Invalid completion count')
        lines.append(f'{title}: {n}/30 ({100*n/30:.2f}% complete)')
    for label, key in [('Dataset', 'current_dataset'), ('Seed', 'current_seed'), ('Method', 'current_variant'),
                       ('Checkpoint', 'latest_checkpoint'), ('Checkpoint time', 'last_checkpoint_timestamp'),
                       ('Warm-up epoch', 'current_warmup_epoch'), ('Online position', 'current_online_observation')]:
        lines.append(f'{label}: {state.get(key)}')
    lines.append('Calibration: ' + str(state.get('calibration', {})))
    lines.append('Last error: ' + str(state.get('last_error')))
    lines.append(state.get('process_note', 'Process activity unknown'))
    lines.append('Counts reflect committed jobs, not elapsed runtime. Status does not write artifacts.')
    return '\n'.join(lines)


def aggregate(out, plan):
    """Called only after v3 checkpoint validation; verify all exported lineage."""
    from .provenance import source_manifest
    out = Path(out)
    if read_json(out / 'protocol.json') != c.protocol() or read_json(out / 'plan.json') != plan:
        raise ValueError('Report protocol mismatch')
    if read_json(out / 'sources.json') != source_manifest():
        raise ValueError('Report source mismatch')
    identities = read_json(out / 'data_identities.json')
    rows = []
    for node in plan['online_jobs']:
        receipt = read_json(out / (node['id'] + '.lineage.json'))
        if receipt['protocol_hash'] != plan['protocol_hash'] or receipt['node_id'] != node['id']:
            raise ValueError('Report lineage mismatch')
        for rel, expected in receipt['artifacts'].items():
            if sha(out / rel) != expected:
                raise ValueError('Report artifact checksum mismatch: ' + rel)
        row = read_json(out / node['id'] / 'result.json')
        if (row['completion_status'] != 'COMPLETE' or row['protocol_hash'] != plan['protocol_hash']
                or row['dataset'] != node['dataset'] or row['seed'] != node['seed']
                or row['variant'] != node['variant'] or row.get('rho_target') != node.get('rho')):
            raise ValueError('Report result identity mismatch')
        identity = identities[c.pair_key(node['dataset'], node['seed'])]
        mask_path, manifest_path = c.mask_paths(out, node['dataset'], node['seed'])
        if sha(mask_path) != identity['mask_artifact_sha256'] or read_json(manifest_path) != identity:
            raise ValueError('Report corruption identity mismatch')
        row.update(data_identity=identity['data_identity'], mask_sha256=identity['mask']['mask_sha256'],
            corruption_fraction=identity['mask']['realized_mask_fraction'],
            effective_changed_value_fraction=identity['mask']['effective_changed_value_fraction'])
        rows.append(row)
    lookup = {(r['dataset'], r['seed'], r['variant']): r for r in rows}
    if len(rows) != 30 or len(lookup) != 30:
        raise ValueError('Report requires all 30 unique completed trajectories')
    metrics = ('MSE', 'MAE', 'online_compute_wall_sec', 'update_fraction', 'updates',
               'eligible_online_steps', 'corruption_fraction', 'effective_changed_value_fraction')
    groups = {}
    for dataset in v3.DATASETS:
        control = [lookup[dataset, seed, 'control'] for seed in v3.SEEDS]
        fc = [lookup[dataset, seed, 'fc'] for seed in v3.SEEDS]
        for a, b in zip(control, fc):
            if a['mask_sha256'] != b['mask_sha256'] or a['data_identity'] != b['data_identity']:
                raise ValueError('Unpaired corruption/scaler')
            if a['MSE'] <= 0 or a['online_compute_wall_sec'] <= 0:
                raise ValueError('Undefined paired ratio')
        groups[dataset] = dict(
            control={k: summary([r[k] for r in control]) for k in metrics},
            fc={k: summary([r[k] for r in fc]) for k in metrics},
            relative_MSE_change=summary([b['MSE']/a['MSE']-1 for a, b in zip(control, fc)]),
            runtime_saving=summary([1-b['online_compute_wall_sec']/a['online_compute_wall_sec'] for a, b in zip(control, fc)]))
    omission = ('Clean-data degradation omitted: no authoritative clean completion/provenance was supplied or validated. '
                'No clean result paths are guessed. MSE here uses each corrupted-TRAIN target-feature scale.')
    report = out / 'reports'
    write_json(report / 'aggregate.json', dict(protocol_hash=plan['protocol_hash'], datasets=groups,
                                              clean_comparison=omission))
    fields = ['dataset', 'seed', 'variant', 'rho_target', *metrics, 'mask_sha256', 'data_identity']
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows({k: r.get(k) for k in fields} for r in rows)
    atomic(report / 'summary.csv', lambda f: f.write(buffer.getvalue()))
    text = '# 30% raw-zero input corruption\n\nClean labels; corrupted TRAIN scaler; rho=1.0; five paired seeds. '
    text += 'Mean ± sample SD (ddof=1). Runtime uses the unchanged v3 online compute boundary. '
    text += 'No rows/timestamps are removed. All historical features are eligible, including ECL series_321.\n\n'
    text += '| Dataset | Method | MSE | MAE | Online seconds | Update fraction | Accepted updates | Eligible steps | Mask fraction | Changed fraction |\n'
    text += '|---|---|' + '---:|' * len(metrics) + '\n'
    for dataset, group in groups.items():
        for method in ('control', 'fc'):
            cells = [f"{group[method][k]['mean']:.6g} ± {group[method][k]['sample_sd']:.6g}" for k in metrics]
            text += f'| {dataset} | {method} | ' + ' | '.join(cells) + ' |\n'
    text += '\nPaired ratios are computed per seed before aggregation.\n'
    for dataset, group in groups.items():
        for name in ('relative_MSE_change', 'runtime_saving'):
            s = group[name]
            text += f"\n{dataset} {name}: {s['mean']:.6g} ± {s['sample_sd']:.6g}.\n"
    text += '\n' + omission + '\n'
    atomic(report / 'REPORT.md', lambda f: f.write(text))
    return {str(p.relative_to(out)): sha(p) for p in (report / 'aggregate.json', report / 'summary.csv', report / 'REPORT.md')}


def show_report(out):
    """Read committed final report without rebuilding or changing anything."""
    out = Path(out)
    path = out / 'completion_manifest.json'
    if not path.exists():
        return 'Final report is available after all 30 trajectories complete; use status for progress.'
    completion = read_json(path)
    if completion['protocol_hash'] != v3.digest(c.protocol()) or completion['online_trajectories'] != 30:
        raise ValueError('Report completion identity mismatch')
    for rel in ('reports/REPORT.md', 'reports/aggregate.json', 'reports/summary.csv'):
        if sha(out / rel) != completion['artifacts'][rel]:
            raise ValueError('Final report checksum mismatch')
    return (out / 'reports/REPORT.md').read_text(encoding='utf-8')


def validate_data():
    """All 15 real data pairs, in temporary storage; no model is instantiated."""
    import tempfile
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from . import runtime
    from .losses import equal
    from .datasets import batch
    before = runtime.rng()
    plan = c.master_plan()
    assert len(plan['online_jobs']) == 30
    for dataset in v3.DATASETS:
        pilot = v3.selection(plan, dataset=dataset, seed=0)
        assert sum(n['kind'] == 'online' for n in pilot) == 2
        assert plan['protocol_hash'] == v3.digest(c.protocol())
    with tempfile.TemporaryDirectory(prefix='odestream-corruption30-validation-') as tmp:
        identities = {}
        corruption_seeds = set()
        for dataset in v3.DATASETS:
            clean = c.read_clean(dataset)
            spec = v3.registry()[dataset]
            target = spec['inputs'].index(spec['target'])
            for seed in v3.SEEDS:
                data, control = c.prepare_pair(tmp, dataset, seed, clean, create=True)
                _, fc = c.prepare_pair(tmp, dataset, seed, clean, windows=False)
                assert control == fc and equal(before, runtime.rng())
                identities[c.pair_key(dataset, seed)] = control
                corruption_seeds.add(control['mask']['corruption_rng_seed'])
                indices, record, _ = c.load_mask(tmp, dataset, seed, clean)
                assert len(indices) == 3 * clean.size // 10
                assert record['realized_mask_fraction'] == (3 * clean.size // 10) / clean.size
                corrupted = clean.copy(order='C')
                corrupted.reshape(-1)[indices] = 0
                independent = StandardScaler().fit(corrupted[:spec['train'][1]])
                np.testing.assert_array_equal(independent.mean_, control['scaler']['mean'])
                np.testing.assert_array_equal(independent.scale_, control['scaler']['scale'])
                scaled = independent.transform(corrupted).astype(np.float32)
                labels = ((clean[:, target] - independent.mean_[target]) / independent.scale_[target]).astype(np.float32)
                selected_target_rows = indices[indices % clean.shape[1] == target] // clean.shape[1]
                changed = selected_target_rows[clean[selected_target_rows, target] != 0]
                assert len(changed) > 0, 'Historical target feature must actually be corrupted'
                for phase, start, end in [('train', 0, spec['train'][1]),
                                          ('validation', spec['train'][1]-24, spec['online'][0]),
                                          ('stream', spec['online'][0]-24, len(clean))]:
                    x, y, t = data[phase]
                    np.testing.assert_array_equal(y[:, 0].numpy(), labels[start+24:end])
                    for offset in range(24):
                        np.testing.assert_array_equal(x[:, offset].numpy(), scaled[start+offset:end-24+offset])
                    # A selected raw target is clean as y and corrupted in the next X.
                    rows = changed[(changed >= start+24) & (changed < end-1)]
                    assert len(rows) > 0
                    row = int(rows[0])
                    assert y[row-start-24, 0].item() == float(labels[row])
                    assert x[row-start-23, -1, target].item() == float(scaled[row, target])
                    assert y[row-start-24, 0].item() != x[row-start-23, -1, target].item()
                    np.testing.assert_array_equal(t[0, :, 0].numpy(), np.linspace(0, 24, 24).astype(np.float32))
                    xb, tb, yb = batch(data[phase], 0)
                    assert xb.shape == (24, 1, spec['input_dim']) and tb.shape == (24, 1, 1) and yb.shape == (1, 1)
                assert equal(before, runtime.rng()), 'Data preparation must not consume any training RNG'
                print(f'{dataset} seed {seed}: paired mask, count, scaler, ALL windows/clean labels, historical target corruption, RNG PASS', flush=True)
                del data
        assert len(corruption_seeds) == 15
        # Exercise the actual inherited identity guard without fabricating model
        # states or invoking the scientific-state validator beyond that guard.
        import copy
        from .runner import Runner
        from .run_corruption30 import CorruptionRunner
        runner = object.__new__(CorruptionRunner)
        runner.out, runner.specs = Path(tmp), v3.registry()
        runner.identities = identities
        runner.hash, runner.source_hash = plan['protocol_hash'], 'validation-source-identity'
        node = plan['nodes'][0]  # Parent-free seed-0 preparation node.
        metadata = runner.metadata(node)
        for key in ('dataset_sha256', 'protocol_hash', 'source_hash', 'corruption', 'node'):
            bad = copy.deepcopy(metadata)
            bad[key] = 'incompatible'
            try:
                Runner.validate(runner, node, bad)
            except ValueError:
                pass
            else:
                raise AssertionError('Checkpoint identity mismatch accepted: ' + key)
        for field in ('mask_sha256', 'mask_artifact_sha256', 'scaler_sha256', 'data_identity'):
            bad = copy.deepcopy(metadata)
            bad['corruption'][field] = 'incompatible'
            try:
                Runner.validate(runner, node, bad)
            except ValueError:
                pass
            else:
                raise AssertionError('Checkpoint corruption mismatch accepted: ' + field)
        assert CorruptionRunner.commit is Runner.commit and CorruptionRunner.finish is Runner.finish
        assert CorruptionRunner.load is Runner.load and CorruptionRunner.coefficient is Runner.coefficient
        # Lost/tampered artifacts must fail, never regenerate under an existing manifest.
        path, _ = c.mask_paths(tmp, 'ECL', 0)
        raw = path.read_bytes()
        path.unlink()
        try:
            c.prepare_pair(tmp, 'ECL', 0, create=True, windows=False)
        except ValueError:
            pass
        else:
            raise AssertionError('Missing authoritative mask regenerated')
        path.write_bytes(raw[:-20])
        try:
            c.prepare_pair(tmp, 'ECL', 0, windows=False)
        except Exception:
            pass
        else:
            raise AssertionError('Damaged mask accepted')
    for path in (v3.DEFAULT_OUTPUT, v3.DEFAULT_OUTPUT / 'nested', v3.ROOT / 'outputs', v3.ROOT / 'experiments'):
        try:
            c.safe_output(path)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsafe output accepted')
    assert equal(before, runtime.rng())
    print('PASS: 15 data pairs; 30 plan entries; checkpoint identity rejection; artifact failure checks; output isolation. No scientific execution.')
