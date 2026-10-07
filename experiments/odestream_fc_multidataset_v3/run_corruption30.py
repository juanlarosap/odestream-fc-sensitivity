"""Separate 30-trajectory runner. plan/status/validate never run scientific work.

Run with python -B -m experiments.odestream_fc_multidataset_v3.run_corruption30.
The subclass reuses v3 scientific dispatch, commits, trace exports, validation,
and checkpoint primitives. No global monkeypatches or clean checkpoint reuse.
"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import socket

from . import config as v3
from . import corruption30 as c
from .checkpointing import now, read_json, sha, write_json
from .provenance import source_manifest, verify_datasets, environment
from .runner import Runner, Interrupted, initial_state


class CorruptionRunner(Runner):
    def __init__(self, out, filters):
        self.out = c.safe_output(out)
        self.filters, self.plan, self.specs = filters, c.master_plan(), v3.registry()
        self.lookup = {n['id']: n for n in self.plan['nodes']}
        self.selected = v3.selection(self.plan, **filters)
        self.hash = self.plan['protocol_hash']
        self.sources = source_manifest()
        self.source_hash = v3.digest(self.sources)
        path = self.out / 'state.json'
        self.state = read_json(path) if path.exists() else initial_state()
        if path.exists() and (self.state['protocol_hash'] != self.hash or self.state['experiment_id'] != c.EXPERIMENT_ID):
            raise ValueError('Corruption master identity mismatch')
        self.state.update(experiment_id=c.EXPERIMENT_ID, protocol_hash=self.hash)
        self.stop_requested, self.current = False, None
        self.data, self.identities, self.active_pair = {}, {}, None

    def write_state(self, **changes):
        self.state.update(changes, last_state_update_timestamp=now())
        for kind, key in [('warmup', 'warmups'), ('online', 'online_trajectories')]:
            n = sum(j['kind'] == kind and j['scientific'] and
                    self.state['jobs'].get(j['id'], {}).get('status') == 'COMPLETE' for j in self.plan['nodes'])
            self.state['completed_' + key], self.state['remaining_' + key] = n, 30 - n
        write_json(self.out / 'state.json', self.state)

    def preflight(self):
        if ((self.out / 'failure.json').exists() or self.state['status'] == 'FAILED'
                or any(j.get('status') == 'FAILED' for j in self.state['jobs'].values())
                or any((self.out / (n['id'] + '.failure.json')).exists() for n in self.plan['nodes'])):
            raise RuntimeError('Durable FAILED marker; investigate manually, no automatic reset')
        verify_datasets()
        for name, value in [('protocol.json', c.protocol()), ('plan.json', self.plan), ('sources.json', self.sources)]:
            self.freeze(self.out / name, value)
        from .runtime import configure
        configure()
        env = environment()
        path = self.out / 'execution_environment.json'
        if path.exists():
            previous = read_json(path)
            for key in ('python_version', 'pytorch', 'numpy', 'pandas', 'sklearn', 'intra_threads', 'inter_threads', 'deterministic', 'device'):
                if previous[key] != env[key]:
                    raise ValueError('Resume environment mismatch: ' + key)
        else:
            write_json(path, env)
        write_json(self.out / 'execution_sessions' / (now().replace(':', '-') + '.json'), env)
        identity_path = self.out / 'data_identities.json'
        frozen = read_json(identity_path) if identity_path.exists() else None
        if frozen is None and any(self.out.rglob('*.pt')):
            raise ValueError('Scientific checkpoints exist without data identities')
        for dataset in v3.DATASETS:
            clean = c.read_clean(dataset)
            for seed in v3.SEEDS:
                self.stop()
                _, manifest = c.prepare_pair(self.out, dataset, seed, clean, create=frozen is None, windows=False)
                self.identities[c.pair_key(dataset, seed)] = manifest
        self.freeze(identity_path, self.identities)
        self.write_state(status='RUNNING', current_stage='RECONCILIATION', pid=os.getpid(),
                         hostname=socket.gethostname(), last_error=None)

    def activate(self, node):
        key = c.pair_key(node['dataset'], node['seed'])
        if self.active_pair != key:
            # One pair in memory; preparation/calibration nodes are always seed 0.
            self.data.clear()
            data, manifest = c.prepare_pair(self.out, node['dataset'], node['seed'])
            if manifest != self.identities[key]:
                raise ValueError('In-memory data identity mismatch')
            self.data[node['dataset']] = data
            self.active_pair = key

    def metadata(self, node):
        manifest = self.identities[c.pair_key(node['dataset'], node['seed'])]
        return dict(super().metadata(node), corruption=dict(data_identity=manifest['data_identity'],
            mask_sha256=manifest['mask']['mask_sha256'], mask_artifact_sha256=manifest['mask_artifact_sha256'],
            scaler_sha256=manifest['scaler_sha256']))

    def check_frozen_identity(self):
        if source_manifest() != self.sources:
            raise ValueError('Scientific sources changed during execution')
        for name, value in [('sources.json', self.sources), ('protocol.json', c.protocol()),
                            ('plan.json', self.plan), ('data_identities.json', self.identities)]:
            if read_json(self.out / name) != value:
                raise ValueError('Frozen identity changed: ' + name)
        for key, manifest in self.identities.items():
            mask, record = c.mask_paths(self.out, manifest['dataset'], manifest['scientific_seed'])
            if sha(mask) != manifest['mask_artifact_sha256'] or read_json(record) != manifest:
                raise ValueError('Frozen corruption artifact changed: ' + key)

    def validate(self, node, ck):
        self.activate(node)
        super().validate(node, ck)

    def run_node(self, node):
        self.activate(node)
        super().run_node(node)

    def reconcile(self):
        # Same v3 recovery semantics, with a 30-job completion contract.
        latest = None
        for node in self.plan['nodes']:
            self.current = node
            old = self.state['jobs'].get(node['id'], {})
            ck = self.load(node)
            if ck is None:
                if old.get('status') == 'COMPLETE' or self.receipt(node).exists() or (self.out / node['id'] / 'result.json').exists():
                    raise ValueError('Missing committed artifact: ' + node['id'])
                self.state['jobs'][node['id']] = dict(status='PENDING')
                continue
            self.validate(node, ck)
            if latest is None or ck['committed_at'] > latest[1]['committed_at']:
                latest = node, ck
            if ck['complete']:
                self.finish(node, ck)
            else:
                if old.get('status') == 'COMPLETE' or self.receipt(node).exists():
                    raise ValueError('Completed job regressed')
                self.state['jobs'][node['id']] = dict(status='INTERRUPTED', artifact=str(self.partial(node).relative_to(self.out)))
            if node['kind'] == 'calibration':
                self.state['calibration'][node['dataset']] = 'COMPLETE' if ck['complete'] else 'INTERRUPTED'
            if node['kind'] == 'lambda':
                self.state['frozen_lambdas'][node['dataset']][str(node['rho'])] = ck['lambda_consistency']
        self.current = None
        if latest:
            self.state.update(**self.projection(*latest))
        path = self.out / 'completion_manifest.json'
        if path.exists():
            record = read_json(path)
            if record['protocol_hash'] != self.hash or record['online_trajectories'] != 30:
                raise ValueError('Completion identity mismatch')
            for rel, expected in record['artifacts'].items():
                if sha(self.out / rel) != expected:
                    raise ValueError('Completion checksum mismatch: ' + rel)
            if any(self.state['jobs'][n['id']]['status'] != 'COMPLETE' for n in self.plan['nodes']):
                raise ValueError('Premature master completion')
        self.write_state()

    def run(self):
        self.preflight()
        self.reconcile()
        for node in self.selected:
            self.stop()
            if self.state['jobs'][node['id']]['status'] != 'COMPLETE':
                self.run_node(node)
        self.check_frozen_identity()
        verify_datasets()
        complete = self.state['completed_online_trajectories'] == 30
        if complete and not (self.out / 'completion_manifest.json').exists():
            from .report_corruption30 import aggregate
            self.write_state(current_stage='REPORTING')
            artifacts = aggregate(self.out, self.plan)
            for node in self.plan['nodes']:
                artifacts.update(read_json(self.receipt(node))['artifacts'])
                artifacts[str(self.receipt(node).relative_to(self.out))] = sha(self.receipt(node))
            for path in [self.out / 'data_identities.json', self.out / 'protocol.json',
                         self.out / 'sources.json', self.out / 'plan.json', *sorted((self.out / 'masks').rglob('*'))]:
                if path.is_file():
                    artifacts[str(path.relative_to(self.out))] = sha(path)
            write_json(self.out / 'completion_manifest.json', dict(protocol_hash=self.hash,
                online_trajectories=30, scientific_warmups=30, completed_at=now(), artifacts=artifacts))
        self.write_state(status='COMPLETE' if complete else 'PENDING',
                         current_stage='COMPLETE' if complete else 'SUBSET_COMPLETE',
                         current_dataset=None, current_seed=None, current_variant=None, current_rho=None)


def execute(out, filters):
    if not __debug__:
        raise RuntimeError('Run without -O; scientific assertions are required')
    out = c.safe_output(out)
    out.mkdir(parents=True, exist_ok=True)
    # Freeze identity before creating the lock, so interrupted startup is recognizable.
    if not (out / 'protocol.json').exists():
        write_json(out / 'protocol.json', c.protocol())
    with (out / 'runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another corruption runner holds the lock') from None
        runner, handlers = None, {}
        try:
            runner = CorruptionRunner(out, filters)
            def requested(signum, frame):
                runner.stop_requested = True
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                handlers[sig] = signal.signal(sig, requested)
            runner.run()
        except (Interrupted, KeyboardInterrupt):
            if runner is not None:
                if runner.current is not None:
                    node = runner.current
                    if runner.state['jobs'].get(node['id'], {}).get('status') != 'COMPLETE':
                        runner.state['jobs'][node['id']] = dict(status='INTERRUPTED')
                    if node['kind'] == 'calibration':
                        runner.state['calibration'][node['dataset']] = 'INTERRUPTED'
                runner.write_state(status='INTERRUPTED', last_error=dict(kind='external_interruption', timestamp=now()))
        except Exception as ex:
            if runner is not None:
                # Persist portable error text, never machine-specific absolute paths.
                message = str(ex).replace(str(v3.ROOT), '.')
                failure = dict(kind='failure', message=message, timestamp=now(),
                    node_id=runner.current['id'] if runner.current else None,
                    recovery='Manual investigation required; no automatic reset, deletion or regeneration')
                if (out / 'failure.json').exists():
                    failure = read_json(out / 'failure.json')
                else:
                    write_json(out / 'failure.json', failure)
                if runner.current:
                    runner.state['jobs'][runner.current['id']] = dict(status='FAILED', error=message)
                runner.write_state(status='FAILED', last_error=failure)
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('plan', 'status', 'run', 'report', 'validate'):
        p = sub.add_parser(command)
        p.add_argument('--output', type=Path, default=Path('outputs') / c.EXPERIMENT_ID)
        if command in ('run', 'plan'):
            p.add_argument('--dataset', choices=v3.DATASETS)
            p.add_argument('--seed', type=int, choices=v3.SEEDS)
    args = parser.parse_args(argv)
    out = c.safe_output(args.output)
    if args.command == 'run':
        return execute(out, dict(dataset=args.dataset, seed=args.seed))
    from .report_corruption30 import status, show_report, validate_data
    if args.command == 'status':
        return print(status(out))
    if args.command == 'validate':
        return validate_data()
    if args.command == 'report':
        return print(show_report(out))
    plan = c.master_plan()
    selected = v3.selection(plan, dataset=args.dataset, seed=args.seed)
    print(c.EXPERIMENT_ID, '\nProtocol hash:', plan['protocol_hash'])
    print('Master: 30 scientific warm-ups + 30 online trajectories; rho=1.0; 3 preparation warm-ups/references')
    for node in selected:
        print(node['id'], '<-', ', '.join(node['parents']) or 'fresh seeded initialization')
    print('Selected online trajectories:', sum(n['kind'] == 'online' for n in selected))
    print('Output:', out.relative_to(v3.ROOT))


if __name__ == '__main__':
    main()
