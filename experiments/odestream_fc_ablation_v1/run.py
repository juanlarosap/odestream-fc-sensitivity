"""Resumable 30-trajectory orchestrator. plan/status/verify do no model work."""
import argparse
import csv
import fcntl
import gzip
import io
import json
import os
from pathlib import Path
import signal
import socket
from experiments.odestream_fc_multidataset_v3 import config as v3
from experiments.odestream_fc_multidataset_v3.checkpointing import (
    atomic, load_checkpoint, now, read_json, save_checkpoint, sha, write_json)
from . import config as c
from .provenance import (source_manifest, verify_datasets, verify_references,
                         environment, check_environment)


class Interrupted(Exception):
    pass


def trace_bytes(rows):
    fields = ['online_index', 'target', 'prediction', 'squared_error', 'absolute_error',
              'update_flag', 'cumulative_update_count', 'gate_threshold', 'rolling_error_statistic', 'diagnostics']
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        row = dict(row, diagnostics=json.dumps(row['diagnostics'], sort_keys=True, allow_nan=False))
        writer.writerow(row)
    return gzip.compress(out.getvalue().encode(), mtime=0)


class Runner:
    def __init__(self, out, v3_output, filters, non_authoritative=False):
        self.out, self.v3_output = Path(out), Path(v3_output).resolve()
        self.plan = c.master_plan()
        self.hash = self.plan['protocol_hash']
        self.lookup = {n['id']: n for n in self.plan['nodes']}
        self.selected = c.selection(self.plan, **filters)
        self.sources = source_manifest()
        self.source_hash = v3.digest(self.sources)
        self.specs = v3.registry()
        self.references, self.data = {}, {}
        self.stop_requested = False
        self.current = None
        self.non_authoritative = non_authoritative
        path = self.out/'state.json'
        self.state = read_json(path) if path.exists() else dict(
            experiment_id=c.EXPERIMENT_ID, protocol_hash=self.hash, status='PENDING', jobs={},
            completed_warmups=0, completed_online_trajectories=0,
            remaining_warmups=30, remaining_online_trajectories=30)
        if self.state['protocol_hash'] != self.hash or self.state['experiment_id'] != c.EXPERIMENT_ID:
            raise ValueError('Ablation state identity mismatch')

    def freeze(self, path, value):
        if path.exists():
            if read_json(path) != value:
                raise ValueError('Frozen identity mismatch: '+str(path))
        else:
            write_json(path, value)

    def path(self, node):
        return self.out/node['id']/'state.pt'

    def receipt(self, node):
        return self.out/(node['id']+'.lineage.json')

    def stop(self, checkpoint=None):
        if self.stop_requested:
            if checkpoint is not None:
                checkpoint()
            raise Interrupted('External signal; last committed state retained')

    def write_state(self, **changes):
        self.state.update(changes, last_state_update_timestamp=now())
        for kind, label in (('warmup', 'warmups'), ('online', 'online_trajectories')):
            count = sum(n['kind'] == kind and self.state['jobs'].get(n['id'], {}).get('status') == 'COMPLETE'
                        for n in self.plan['nodes'])
            self.state['completed_'+label] = count
            self.state['remaining_'+label] = 30-count
        write_json(self.out/'state.json', self.state)

    def preflight(self):
        if (self.out/'failure.json').exists() or self.state['status'] == 'FAILED':
            raise RuntimeError('Durable FAILED state; investigate before resuming')
        if any(n.get('status') == 'FAILED' for n in self.state['jobs'].values()):
            raise RuntimeError('FAILED job blocks execution')
        verify_datasets()
        needed = {n['dataset'] for n in self.selected if n['fc_enabled']}
        self.references = verify_references(self.v3_output, [d for d in c.DATASETS if d in needed])
        from experiments.odestream_fc_multidataset_v3.runtime import configure
        configure()
        env = environment()
        for record in self.references.values():
            check_environment(record['execution_environment'], env)
        self.freeze(self.out/'protocol.json', c.protocol())
        self.freeze(self.out/'plan.json', self.plan)
        self.freeze(self.out/'sources.json', self.sources)
        self.freeze(self.out/'execution_identity.json', dict(non_authoritative=self.non_authoritative))
        path = self.out/'execution_environment.json'
        if path.exists():
            check_environment(read_json(path), env)
        else:
            write_json(path, env)
        write_json(self.out/'execution_sessions'/(now().replace(':','-')+'.json'),
                   dict(env, v3_output=str(self.v3_output), non_authoritative=self.non_authoritative))
        for dataset, record in self.references.items():
            self.freeze(self.out/'datasets'/dataset/'calibration_reference.json', record)
        from experiments.odestream_fc_multidataset_v3.datasets import load_data
        for dataset in c.DATASETS:
            if any(n['dataset'] == dataset for n in self.selected):
                self.stop()
                data, manifest = load_data(dataset)
                self.data[dataset] = data
                self.freeze(self.out/'datasets'/dataset/'dataset_manifest.json', manifest)
        self.write_state(status='RUNNING', pid=os.getpid(), hostname=socket.gethostname(), last_error=None)

    def check_frozen_identity(self):
        for name, expected in (('protocol.json', c.protocol()), ('plan.json', self.plan), ('sources.json', self.sources)):
            if read_json(self.out/name) != expected:
                raise ValueError('Frozen identity changed: '+name)
        if source_manifest() != self.sources:
            raise ValueError('Scientific source changed during execution')
        for dataset, record in self.references.items():
            if read_json(self.out/'datasets'/dataset/'calibration_reference.json') != record:
                raise ValueError('Frozen calibration reference changed')
            # Small coefficient/reference metadata checks each commit; full ancestry at preflight/end.
            for rel, checksum in record['artifacts'].items():
                if rel.endswith('.json') and sha(self.v3_output/rel) != checksum:
                    raise ValueError('Referenced v3 calibration artifact changed: '+rel)

    def coefficient(self, node):
        if not node['fc_enabled']:
            return None, {}
        record = self.references[node['dataset']]
        return record['lambda_consistency'], dict(calibration_identity=record['coefficient_sha256'], rho_target=1.0)

    def metadata(self, node):
        parents = {}
        for identifier in node['parents']:
            parent = self.lookup[identifier]
            path = self.path(parent)
            receipt = read_json(self.receipt(parent))
            rel = str(path.relative_to(self.out))
            checksum = sha(path)
            if (receipt.get('protocol_hash') != self.hash or receipt.get('node_id') != identifier
                    or receipt['artifacts'].get(rel) != checksum):
                raise ValueError('Parent lineage mismatch: '+identifier)
            parents[rel] = checksum
        record = self.references.get(node['dataset']) if node['fc_enabled'] else None
        return dict(protocol_hash=self.hash, source_hash=self.source_hash, node=node,
                    dataset_sha256=self.specs[node['dataset']]['sha256'], parents=parents,
                    calibration_reference_hash=v3.digest(record) if record else None)

    def load(self, node):
        return load_checkpoint(self.path(node)) if self.path(node).exists() else None

    def validate(self, node, ck):
        from . import engine
        from experiments.odestream_fc_multidataset_v3 import engine as base
        from experiments.odestream_fc_multidataset_v3.losses import finite
        for key, value in self.metadata(node).items():
            if ck.get(key) != value:
                raise ValueError('Checkpoint identity mismatch: '+node['id']+'/'+key)
        internal, spec = c.internal_node(node), self.specs[node['dataset']]
        if node['kind'] == 'warmup':
            base.validate_warmup(ck, internal, spec)
        elif node['kind'] == 'bootstrap':
            if node['gate_mode'] != 'r50' or not ck['complete'] or len(ck['rows']) != 256 or len(ck['errors']) != 256:
                raise ValueError('Invalid bootstrap')
            for i, (row, error) in enumerate(zip(ck['rows'], ck['errors'])):
                if (row['dataset_row'] != spec['bootstrap_targets'][0]+i or error != row['squared_error']
                        or error != (row['target']-row['prediction'])**2):
                    raise ValueError('Bootstrap error mismatch')
                if node['dataset'] in self.data and row['target'] != self.data[node['dataset']]['validation'][1][-256+i].item():
                    raise ValueError('Bootstrap target mismatch')
            finite(ck['rows'], 'bootstrap')
        elif node['kind'] == 'online':
            boot = self.load(self.lookup[node['parents'][1]])['errors'] if node['gate_mode'] == 'r50' else []
            engine.validate_online(ck, node, spec, boot, self.data.get(node['dataset']))
        else:
            raise ValueError('Unknown stage')
        if node['fc_enabled'] and node['kind'] in ('warmup', 'online'):
            coefficient, provenance = self.coefficient(node)
            state = ck['state']['method']
            if state['lambda_consistency'] != coefficient or state['provenance'] != provenance:
                raise ValueError('FC coefficient/provenance mismatch')

    def finish(self, node, ck):
        path, receipt_path = self.path(node), self.receipt(node)
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt['protocol_hash'] != self.hash or receipt['node_id'] != node['id']:
                raise ValueError('Lineage identity mismatch')
            for rel, checksum in receipt['artifacts'].items():
                if sha(self.out/rel) != checksum:
                    raise ValueError('Completed artifact checksum mismatch: '+rel)
        artifacts = {str(path.relative_to(self.out)): sha(path)}
        if node['kind'] == 'online':
            from .engine import result
            coefficient_hash = self.references[node['dataset']]['coefficient_sha256'] if node['fc_enabled'] else None
            value = result(node, ck, self.hash, sha(self.path(self.lookup[node['parents'][0]])), coefficient_hash)
            value.update(source_hash=self.source_hash,
                         dataset_sha256=self.specs[node['dataset']]['sha256'],
                         non_authoritative=self.non_authoritative)
            dest = self.out/node['id']/'result.json'
            self.freeze(dest, value)
            trace = self.out/node['id']/'trace.csv.gz'
            raw = trace_bytes(ck['rows'])
            if receipt_path.exists():
                if trace.read_bytes() != raw:
                    raise ValueError('Completed trace/checkpoint mismatch')
            else:
                atomic(trace, lambda f: f.write(raw), True)
            artifacts.update({str(p.relative_to(self.out)): sha(p) for p in (dest, trace)})
        record = dict(protocol_hash=self.hash, node_id=node['id'], parents=ck['parents'], artifacts=artifacts)
        self.freeze(receipt_path, record)
        self.state['jobs'][node['id']] = dict(status='COMPLETE', artifact=str(path.relative_to(self.out)), sha256=sha(path))

    def commit(self, node, value):
        self.check_frozen_identity()
        value = dict(value, **self.metadata(node), committed_at=now())
        self.validate(node, value)
        save_checkpoint(self.path(node), value)
        if node['kind'] == 'online':
            raw = trace_bytes(value['rows'])
            atomic(self.out/node['id']/'trace.csv.gz', lambda f: f.write(raw), True)
        if value['complete']:
            self.finish(node, value)
        self.write_state(current_node=node['id'], current_stage=node['kind'], latest_checkpoint=str(self.path(node)),
                         current_epoch=value.get('epoch'), current_batch=value.get('batch_index'),
                         current_online_observation=value.get('next_index'), accepted_updates=value.get('updates'))
        return value

    def reconcile(self):
        selected_ids = {n['id'] for n in self.selected}
        for node in self.plan['nodes']:
            self.current = node
            old = self.state['jobs'].get(node['id'], {})
            # An R50-only invocation never loads FC coefficients, even when sharing
            # an output root with completed FC work. Check unselected receipts as
            # bytes; scientific validation occurs when selected or before reporting.
            if node['id'] not in selected_ids:
                if self.receipt(node).exists():
                    receipt = read_json(self.receipt(node))
                    if receipt['protocol_hash'] != self.hash or receipt['node_id'] != node['id']:
                        raise ValueError('Unselected lineage identity mismatch')
                    required = {str(self.path(node).relative_to(self.out))}
                    if node['kind'] == 'online':
                        required.update(node['id']+'/'+name for name in ('result.json', 'trace.csv.gz'))
                    if set(receipt['artifacts']) != required:
                        raise ValueError('Unselected lineage artifact inventory mismatch')
                    for rel, checksum in receipt['artifacts'].items():
                        if sha(self.out/rel) != checksum:
                            raise ValueError('Unselected artifact checksum mismatch: '+rel)
                    self.state['jobs'][node['id']] = dict(status='COMPLETE')
                elif old.get('status') == 'COMPLETE':
                    raise ValueError('Missing completed lineage: '+node['id'])
                else:
                    self.state['jobs'][node['id']] = dict(status='INTERRUPTED' if self.path(node).exists() else 'PENDING')
                continue
            ck = self.load(node)
            if ck is None:
                if old.get('status') == 'COMPLETE' or self.receipt(node).exists() or (self.out/node['id']/'result.json').exists():
                    raise ValueError('Missing checkpoint for existing result: '+node['id'])
                self.state['jobs'][node['id']] = dict(status='PENDING')
                continue
            self.validate(node, ck)
            if ck['complete']:
                self.finish(node, ck)
            else:
                if old.get('status') == 'COMPLETE' or self.receipt(node).exists():
                    raise ValueError('Completed job regressed')
                self.state['jobs'][node['id']] = dict(status='INTERRUPTED')
        completion = self.out/'completion_manifest.json'
        if completion.exists():
            manifest = read_json(completion)
            if manifest['protocol_hash'] != self.hash or manifest['online_trajectories'] != 30:
                raise ValueError('Completion identity mismatch')
            for rel, checksum in manifest['artifacts'].items():
                if sha(self.out/rel) != checksum:
                    raise ValueError('Completion artifact mismatch: '+rel)
            if any(self.state['jobs'][n['id']]['status'] != 'COMPLETE' for n in self.plan['nodes']):
                raise ValueError('Premature completion manifest')
        self.current = None
        self.write_state()

    def run_node(self, node):
        from . import engine
        saved = self.load(node)
        if saved is not None:
            self.validate(node, saved)
            if saved['complete']:
                self.finish(node, saved)
                return
        self.stop()
        self.current = node
        self.state['jobs'][node['id']] = dict(status='RUNNING')
        self.write_state(current_node=node['id'], current_stage=node['kind'])
        commit = lambda value: self.commit(node, value)
        data, spec = self.data[node['dataset']], self.specs[node['dataset']]
        coefficient, provenance = self.coefficient(node)
        if node['kind'] == 'warmup':
            engine.warmup(c.internal_node(node), spec, data, coefficient, provenance, saved, commit, self.stop)
        elif node['kind'] == 'bootstrap':
            engine.bootstrap(c.internal_node(node), spec, data, self.load(self.lookup[node['parents'][0]]), commit, self.stop)
        elif node['kind'] == 'online':
            initial = self.load(self.lookup[node['parents'][0]])
            boot = self.load(self.lookup[node['parents'][1]])['errors'] if node['gate_mode'] == 'r50' else []
            engine.online(node, spec, data, initial, boot, coefficient, provenance, saved, commit, self.stop)
        else:
            raise ValueError('No calibration or other conditions may execute')
        self.current = None

    def run(self):
        self.preflight()
        self.reconcile()
        for node in self.selected:
            if self.state['jobs'][node['id']]['status'] != 'COMPLETE':
                self.run_node(node)
        self.check_frozen_identity()
        verify_datasets()
        if verify_references(self.v3_output, list(self.references)) != self.references:
            raise ValueError('Calibration ancestry changed during run')
        if self.state['completed_online_trajectories'] == 30:
            if not (self.out/'completion_manifest.json').exists():
                from .reporting import aggregate
                artifacts = aggregate(self.out)
                for node in self.plan['nodes']:
                    receipt = read_json(self.receipt(node))
                    artifacts.update(receipt['artifacts'])
                    artifacts[str(self.receipt(node).relative_to(self.out))] = sha(self.receipt(node))
                for path in (self.out/'protocol.json', self.out/'plan.json', self.out/'sources.json',
                             self.out/'execution_identity.json', self.out/'execution_environment.json'):
                    artifacts[str(path.relative_to(self.out))] = sha(path)
                for path in (self.out/'datasets').glob('*/calibration_reference.json'):
                    artifacts[str(path.relative_to(self.out))] = sha(path)
                for path in (self.out/'datasets').glob('*/dataset_manifest.json'):
                    artifacts[str(path.relative_to(self.out))] = sha(path)
                write_json(self.out/'completion_manifest.json', dict(protocol_hash=self.hash,
                           source_hash=self.source_hash, online_trajectories=30, scientific_warmups=30,
                           non_authoritative=self.non_authoritative, completed_at=now(), artifacts=artifacts))
            status = 'COMPLETE'
        else:
            status = 'SUBSET_COMPLETE'
        self.write_state(status=status, current_node=None, current_stage=status)


def execute(out, v3_output, filters, non_authoritative=False):
    if not __debug__:
        raise RuntimeError('Do not run scientific invariants with Python -O')
    out = c.safe_output(out, v3_output)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another ablation runner holds the lock')
        runner, handlers = None, {}
        try:
            runner = Runner(out, v3_output, filters, non_authoritative)
            def requested(signum, frame):
                runner.stop_requested = True
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                handlers[sig] = signal.signal(sig, requested)
            runner.run()
        except (Interrupted, KeyboardInterrupt) as exc:
            if runner is not None:
                if runner.current is not None and runner.state['jobs'].get(runner.current['id'], {}).get('status') != 'COMPLETE':
                    runner.state['jobs'][runner.current['id']] = dict(status='INTERRUPTED')
                runner.write_state(status='INTERRUPTED', last_error=str(exc))
        except Exception as exc:
            if runner is not None:
                if runner.current is not None:
                    runner.state['jobs'][runner.current['id']] = dict(status='FAILED')
                failure = dict(error=repr(exc), node=runner.current, recorded_at=now(),
                               recovery='Investigate; no automatic reset or checkpoint deletion')
                if not (out/'failure.json').exists():
                    write_json(out/'failure.json', failure)
                runner.write_state(status='FAILED', last_error=failure)
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    for name in ('plan', 'verify', 'run', 'status'):
        sub = subs.add_parser(name)
        sub.add_argument('--output', type=Path, default=c.DEFAULT_OUTPUT)
        if name != 'status':
            sub.add_argument('--dataset', choices=c.DATASETS)
            sub.add_argument('--seed', type=int, choices=c.SEEDS)
            sub.add_argument('--conditions', nargs='+', choices=list(c.CONDITIONS), default=list(c.CONDITIONS))
            sub.add_argument('--v3-output', type=Path, default=c.DEFAULT_V3_OUTPUT)
        if name == 'plan':
            sub.add_argument('--json', action='store_true')
        if name == 'run':
            sub.add_argument('--non-authoritative', action='store_true', help='Mark separate local validation artifacts')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'status':
        path = args.output/'state.json'
        print(json.dumps(read_json(path) if path.exists() else dict(status='NOT_STARTED'), indent=2))
        return
    filters = dict(dataset=args.dataset, seed=args.seed, conditions=args.conditions)
    plan = c.master_plan()
    selected = c.selection(plan, **filters)
    if args.command == 'plan':
        if args.json:
            print(json.dumps(dict(master=plan, selected_ids=[n['id'] for n in selected]), indent=2))
        else:
            print('Master: 30 new scientific online trajectories; 30 independent warmups; 15 R50 bootstraps; 0 calibrations.')
            print('Only r50_only and fc_only. FC rho_target is fixed at 1.0.')
            print('Selected online trajectories:', sum(n['kind'] == 'online' for n in selected))
            for n in selected:
                print(n['id'], 'FC='+str(n['fc_enabled']), 'gate='+n['gate_mode'])
        return
    if args.command == 'verify':
        verify_datasets()
        datasets = [d for d in c.DATASETS if any(n['dataset'] == d and n['fc_enabled'] for n in selected)]
        records = verify_references(args.v3_output, datasets)
        from experiments.odestream_fc_multidataset_v3.runtime import configure
        configure()
        current_environment = environment()
        for record in records.values():
            check_environment(record['execution_environment'], current_environment)
        print(json.dumps(dict(status='VERIFIED', scientific_execution=False,
                              current_execution_environment=current_environment,
                              calibration={d: dict(rho_target=1.0, lambda_consistency=r['lambda_consistency'],
                                  coefficient_sha256=r['coefficient_sha256'],
                                  execution_environment=r['execution_environment']) for d, r in records.items()},
                              sources=source_manifest()), indent=2))
        return
    execute(args.output, args.v3_output, filters, args.non_authoritative)


if __name__ == '__main__':
    main()
