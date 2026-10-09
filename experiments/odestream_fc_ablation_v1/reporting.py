"""Standalone five-seed reports for the two new conditions; no old results read."""
import argparse
import csv
import fcntl
import io
from pathlib import Path
from experiments.odestream_fc_multidataset_v3 import config as v3
from experiments.odestream_fc_multidataset_v3.checkpointing import atomic, read_json, sha, write_json
from experiments.odestream_fc_multidataset_v3.reporting import summary
from . import config as c
from .provenance import source_manifest

METRICS = ('MSE', 'MAE', 'online_compute_wall_sec', 'update_fraction', 'updates', 'eligible_online_steps')


def collect(out):
    """Read-only checkpoint/receipt/result validation before any report writes."""
    from .run import Runner, trace_bytes
    from .engine import result
    out = Path(out)
    if (out/'failure.json').exists():
        raise ValueError('Cannot report a failed experiment')
    if read_json(out/'protocol.json') != c.protocol() or read_json(out/'plan.json') != c.master_plan():
        raise ValueError('Report protocol/plan mismatch')
    if read_json(out/'sources.json') != source_manifest():
        raise ValueError('Report source mismatch')
    non_authoritative = read_json(out/'execution_identity.json')['non_authoritative']
    runner = Runner(out, c.DEFAULT_V3_OUTPUT, {}, non_authoritative)
    for dataset in c.DATASETS:
        runner.references[dataset] = read_json(out/'datasets'/dataset/'calibration_reference.json')
        manifest = read_json(out/'datasets'/dataset/'dataset_manifest.json')
        if any(manifest.get(k) != v for k, v in v3.registry()[dataset].items()):
            raise ValueError('Dataset manifest mismatch')
    rows = []
    for node in runner.plan['nodes']:
        ck = runner.load(node)
        if ck is None or not ck['complete']:
            raise ValueError('Report requires all 30 trajectories: incomplete '+node['id'])
        runner.validate(node, ck)
        receipt = read_json(runner.receipt(node))
        paths = [runner.path(node)]
        if node['kind'] == 'online':
            paths += [out/node['id']/'result.json', out/node['id']/'trace.csv.gz']
        expected = dict(protocol_hash=runner.hash, node_id=node['id'], parents=ck['parents'],
                        artifacts={str(p.relative_to(out)): sha(p) for p in paths})
        if receipt != expected:
            raise ValueError('Report lineage mismatch: '+node['id'])
        if node['kind'] == 'online':
            coefficient_hash = runner.references[node['dataset']]['coefficient_sha256'] if node['fc_enabled'] else None
            value = result(node, ck, runner.hash, sha(runner.path(runner.lookup[node['parents'][0]])), coefficient_hash)
            value.update(source_hash=runner.source_hash, dataset_sha256=v3.registry()[node['dataset']]['sha256'],
                         non_authoritative=non_authoritative)
            if read_json(paths[1]) != value or paths[2].read_bytes() != trace_bytes(ck['rows']):
                raise ValueError('Result/trace differs from checkpoint')
            rows.append(value)
    completion = out/'completion_manifest.json'
    if completion.exists():
        record = read_json(completion)
        if record['protocol_hash'] != runner.hash or record['online_trajectories'] != 30:
            raise ValueError('Completion identity mismatch')
        for rel, checksum in record['artifacts'].items():
            if sha(out/rel) != checksum:
                raise ValueError('Completion checksum mismatch: '+rel)
    return rows


def aggregate(out):
    out = Path(out)
    rows = collect(out)
    lookup = {(r['dataset'], r['condition'], r['seed']): r for r in rows}
    if len(lookup) != 30:
        raise ValueError('Expected 30 distinct results')
    groups = {}
    for dataset in c.DATASETS:
        groups[dataset] = {}
        for name in c.CONDITIONS:
            values = [lookup[dataset, name, seed] for seed in c.SEEDS]
            groups[dataset][name] = {key: summary([r[key] for r in values]) for key in METRICS}
    report = out/'reports'
    write_json(report/'aggregate.json', dict(experiment_id=c.EXPERIMENT_ID, seed_order=list(c.SEEDS),
               non_authoritative=rows[0]['non_authoritative'], datasets=groups))
    fields = ['dataset', 'condition', 'seed', 'fc_enabled', 'gate_mode', 'rho_target',
              'lambda_consistency', *METRICS, 'non_authoritative']
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows({k: r.get(k) for k in fields} for r in rows)
    atomic(report/'summary.csv', lambda f: f.write(buf.getvalue()))
    text = '# ODEStream ablation: missing conditions\n\n'
    if rows[0]['non_authoritative']:
        text += '**NON-AUTHORITATIVE LOCAL VALIDATION**\n\n'
    text += ('Exactly five seeds per dataset/condition. Mean ± sample SD (ddof=1). '
             'MSE/MAE use TRAIN-standardized targets. Time is the original v3 online compute boundary, '
             'including built-in diagnostics. FC includes warmup and online treatment. '
             'No Control-relative metrics or old scientific results are imported. '
             'JSON includes min/max and individual values in seed order 0–4; CSV contains each seed.\n\n')
    text += '| Dataset | Condition | MSE | MAE | Online seconds | Update fraction | Accepted updates | Eligible steps |\n'
    text += '|---|---|---:|---:|---:|---:|---:|---:|\n'
    for dataset in c.DATASETS:
        for name in c.CONDITIONS:
            metrics = groups[dataset][name]
            cells = [f"{metrics[k]['mean']:.9g} ± {metrics[k]['sample_sd']:.9g}" for k in METRICS]
            text += '| '+ ' | '.join([dataset, name, *cells])+' |\n'
    atomic(report/'REPORT.md', lambda f: f.write(text))
    return {str(p.relative_to(out)): sha(p) for p in
            (report/'aggregate.json', report/'summary.csv', report/'REPORT.md')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=c.DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    out = c.safe_output(args.output)
    if not (out/'protocol.json').exists():
        raise ValueError('No ablation output exists')
    with (out/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        aggregate(out)
    print('Reports verified and written to', out/'reports')


if __name__ == '__main__':
    main()
