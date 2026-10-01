"""Idempotent master orchestrator. Only execute() starts scientific work."""
import copy
import csv
import fcntl
import gzip
import io
import json
import os
from pathlib import Path
import signal
import socket
from .config import protocol, digest, master_plan, selection, registry, artifact_path, EXPERIMENT_ID
from .checkpointing import now, sha, read_json, write_json, atomic, save_checkpoint, load_checkpoint
from .provenance import verify_datasets, source_manifest, environment


class Interrupted(Exception):pass


def initial_state():
    return dict(experiment_id=EXPERIMENT_ID,protocol_hash=digest(protocol()),status='PENDING',
        current_stage='PREFLIGHT',current_dataset=None,current_seed=None,current_rho=None,current_variant=None,
        current_warmup_epoch=None,current_warmup_batch=None,current_warmup_batch_total=None,
        current_online_observation=0,current_online_total=None,accepted_updates=0,realized_update_fraction=None,
        latest_checkpoint=None,completed_warmups=0,completed_online_trajectories=0,
        remaining_warmups=90,remaining_online_trajectories=90,
        calibration={d:'PENDING' for d in registry()},frozen_lambdas={d:{} for d in registry()},
        last_checkpoint_timestamp=None,last_state_update_timestamp=None,last_error=None,jobs={})


class Runner:
    def __init__(self,out,filters):
        self.out=Path(out);self.filters=filters;self.plan=master_plan();self.specs=registry()
        self.lookup={j['id']:j for j in self.plan['nodes']};self.selected=selection(self.plan,**filters)
        self.hash=self.plan['protocol_hash'];self.sources=source_manifest();self.source_hash=digest(self.sources)
        path=self.out/'state.json';self.state=read_json(path) if path.exists() else initial_state()
        if self.state['protocol_hash']!=self.hash:raise ValueError('Master state protocol mismatch')
        self.stop_requested=False;self.current=None;self.data={}
    def write_state(self,**changes):
        self.state.update(changes,last_state_update_timestamp=now())
        jobs=self.state['jobs']
        for kind,key in [('warmup','warmups'),('online','online_trajectories')]:
            n=sum(j['kind']==kind and j['scientific'] and jobs.get(j['id'],{}).get('status')=='COMPLETE' for j in self.plan['nodes'])
            self.state['completed_'+key]=n;self.state['remaining_'+key]=90-n
        write_json(self.out/'state.json',self.state)
    def stop(self,checkpoint=None):
        if self.stop_requested:
            if checkpoint is not None:checkpoint()
            raise Interrupted('External signal; last valid committed state retained')
    def freeze(self,path,value):
        if path.exists():
            if read_json(path)!=value:raise ValueError('Frozen identity mismatch: '+str(path))
        else:write_json(path,value)
    def preflight(self):
        if (self.out/'failure.json').exists():raise RuntimeError('Durable FAILED marker blocks execution; manual investigation required')
        if any((self.out/(j['id']+'.failure.json')).exists() for j in self.plan['nodes']):
            raise RuntimeError('Durable FAILED job marker blocks execution; manual investigation required')
        if self.state['status']=='FAILED' or any(j['status']=='FAILED' for j in self.state['jobs'].values()):
            raise RuntimeError('FAILED experiment blocks execution; manual investigation required')
        verify_datasets()
        self.freeze(self.out/'protocol.json',protocol())
        self.freeze(self.out/'plan.json',self.plan)
        self.freeze(self.out/'sources.json',self.sources)
        from .runtime import configure
        configure()
        env=environment();path=self.out/'execution_environment.json'
        if path.exists():
            previous=read_json(path)
            # Host/path/OS metadata never enters scientific identity. Numerical stack
            # changes on an existing execution require investigation, not silent replay.
            for k in ('python_version','pytorch','numpy','pandas','sklearn','intra_threads','inter_threads','deterministic','device'):
                if previous[k]!=env[k]:raise ValueError('Resume execution compatibility mismatch: '+k)
        else:write_json(path,env)
        write_json(self.out/'execution_sessions'/(now().replace(':','-')+'.json'),env)
        from .datasets import load_data
        for d in self.specs:
            self.stop();data,manifest=load_data(d);self.data[d]=data
            self.freeze(self.out/'datasets'/d/'dataset_manifest.json',manifest)
        self.write_state(status='RUNNING',current_stage='RECONCILIATION',pid=os.getpid(),hostname=socket.gethostname(),last_error=None)
    def path(self,node):return artifact_path(self.out,node)
    def partial(self,node):return self.out/(node['id']+'.state.pt') if node['kind']=='calibration' else self.path(node)
    def receipt(self,node):return self.out/(node['id']+'.lineage.json')
    def parents(self,node):
        parents={}
        for key in node['parents']:
            parent=self.lookup[key];path=self.path(parent)
            if not path.exists():raise ValueError('Missing parent artifact: '+key)
            value=read_json(path) if parent['kind'] in ('calibration','lambda') else load_checkpoint(path)
            if not value['complete']:raise ValueError('Incomplete parent artifact: '+key)
            if value['protocol_hash']!=self.hash or value['source_hash']!=self.source_hash or value['node']!=parent:
                raise ValueError('Parent identity mismatch: '+key)
            rel=str(path.relative_to(self.out));checksum=sha(path)
            receipt=read_json(self.receipt(parent))
            if receipt['artifacts'].get(rel)!=checksum:raise ValueError('Parent lineage checksum mismatch: '+key)
            parents[rel]=checksum
        return parents
    def metadata(self,node):
        return dict(protocol_hash=self.hash,source_hash=self.source_hash,node=node,
                    dataset_sha256=self.specs[node['dataset']]['sha256'],parents=self.parents(node))
    def check_frozen_identity(self):
        if source_manifest()!=self.sources:raise ValueError('Canonical source changed during execution')
        if read_json(self.out/'sources.json')!=self.sources:raise ValueError('Source manifest changed')
        if read_json(self.out/'protocol.json')!=protocol():raise ValueError('Protocol changed during execution')
        if read_json(self.out/'plan.json')!=self.plan:raise ValueError('Master plan changed during execution')
    def load(self,node):
        path=self.path(node)
        if path.exists():return read_json(path) if node['kind'] in ('calibration','lambda') else load_checkpoint(path)
        path=self.partial(node)
        return load_checkpoint(path) if path.exists() else None
    def lambda_record(self,node):
        if node['variant']!='fc':return None
        return read_json(self.out/f"datasets/{node['dataset']}/calibration/rho_{node['rho']}.json")
    def coefficient(self,node):
        record=self.lambda_record(node)
        if record is None:return None,{}
        path=self.out/f"datasets/{node['dataset']}/calibration/rho_{node['rho']}.json"
        return record['lambda_consistency'],dict(calibration_identity=sha(path),rho_target=node['rho'])
    def validate(self,node,ck):
        expected=self.metadata(node)
        for k,v in expected.items():
            if ck[k]!=v:raise ValueError('Artifact identity mismatch: '+node['id']+' / '+k)
        from . import engine
        from .losses import finite, equal
        spec=self.specs[node['dataset']]
        if node['kind']=='warmup':engine.validate_warmup(ck,node,spec)
        elif node['kind']=='online':
            boot=self.load(self.lookup[node['parents'][1]])['errors'] if node['variant']=='fc' else []
            engine.validate_online(ck,node,spec,boot,self.data.get(node['dataset']))
        elif node['kind']=='bootstrap':
            if not ck['complete'] or len(ck['rows'])!=256 or len(ck['errors'])!=256:raise ValueError('Invalid bootstrap')
            for i,(row,error) in enumerate(zip(ck['rows'],ck['errors'])):
                if row['dataset_row']!=spec['bootstrap_targets'][0]+i or error!=row['squared_error'] or error!=(row['target']-row['prediction'])**2:raise ValueError('Bootstrap error mismatch')
                if node['dataset'] in self.data and row['target']!=self.data[node['dataset']]['validation'][1][-256+i].item():raise ValueError('Bootstrap target mismatch')
            finite(ck['rows'],'bootstrap')
        elif node['kind']=='calibration':
            if ck['complete']:
                if ck['steps']!=128 or len(ck['rows'])!=128 or not ck['all_128_baseline_commits_exact']:raise ValueError('Invalid calibration completion')
                import numpy as np
                valid=[r['unscaled_gradient_ratio'] for r in ck['rows'] if r['G_cons']>0]
                if len(valid)<2 or ck['median_unscaled_gradient_ratio']!=float(np.median(valid)) or ck['median_unscaled_gradient_ratio']<=0:raise ValueError('Invalid calibration median')
                finite(ck['rows'],'calibration');finite(ck['median_unscaled_gradient_ratio'],'calibration median')
            else:
                fc=ck['fc'];n=ck['next_index']
                if not 0<=n<=128 or fc['student_update_count']!=n or fc['teacher_update_count']!=n or len(fc['rows'])!=n:raise ValueError('Calibration position mismatch')
                if fc['lambda_consistency']!=0 or fc['beta']!=.99 or fc['level']!='calibration':raise ValueError('Calibration contract mismatch')
                state=dict(model=fc['student'],method=None,optimizer=ck['optimizer'],rng=ck['rng'])
                engine.validate_state(state,'control',n);finite(fc,'calibration state')
        elif node['kind']=='lambda':
            from .calibration import derive
            reference=self.load(self.lookup[node['parents'][0]])
            wanted=derive(reference,node['rho'])
            if any(ck[k]!=v for k,v in wanted.items()):raise ValueError('Lambda derivation mismatch')
        else:raise ValueError('Unexpected stage')
        if node['variant']=='fc' and node['kind'] in ('warmup','online'):
            coefficient,provenance=self.coefficient(node)
            method=ck['state']['method']
            if method['lambda_consistency']!=coefficient or method['provenance']!=provenance:raise ValueError('FC calibration mismatch')
    def projection(self,node,ck):
        path=self.path(node) if ck['complete'] else self.partial(node)
        changes=dict(current_dataset=node['dataset'],current_seed=node['seed'],current_rho=node.get('rho'),
            current_variant=node['variant'],current_stage=node['kind'].upper(),latest_checkpoint=str(path.relative_to(self.out)),
            last_checkpoint_timestamp=ck['committed_at'],current_warmup_epoch=None,current_warmup_batch=None,
            current_warmup_batch_total=None,current_online_observation=0,current_online_total=None,
            accepted_updates=0,realized_update_fraction=None)
        if node['kind']=='warmup':changes.update(current_warmup_epoch=ck['epoch'],current_warmup_batch=ck['batch_index'],
            current_warmup_batch_total=self.specs[node['dataset']]['train_windows']//64)
        if node['kind']=='online':changes.update(current_online_observation=ck['next_index'],current_online_total=ck['n'],
            accepted_updates=ck['updates'],realized_update_fraction=ck['updates']/ck['next_index'] if ck['next_index'] else None)
        if node['kind']=='calibration':self.state['calibration'][node['dataset']]='COMPLETE' if ck['complete'] else 'RUNNING'
        if node['kind']=='lambda':self.state['frozen_lambdas'][node['dataset']][str(node['rho'])]=ck['lambda_consistency']
        return changes
    def trace_bytes(self,rows):
        fields=['online_index','target','prediction','squared_error','absolute_error','update_flag',
                'cumulative_update_count','gate_threshold','rolling_error_statistic','diagnostics']
        out=io.StringIO();writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
        for row in rows:
            row=copy.deepcopy(row);row['diagnostics']=json.dumps(row['diagnostics'],sort_keys=True,allow_nan=False);writer.writerow(row)
        return gzip.compress(out.getvalue().encode(),mtime=0)
    def finish(self,node,ck):
        """Recover derived exports after a crash; never repeat completed science."""
        path=self.path(node);receipt=self.receipt(node)
        if receipt.exists():
            record=read_json(receipt)
            if record['protocol_hash']!=self.hash or record['node_id']!=node['id']:raise ValueError('Lineage identity mismatch')
            for rel,digest_value in record['artifacts'].items():
                if sha(self.out/rel)!=digest_value:raise ValueError('Completed artifact checksum mismatch: '+rel)
        artifacts={str(path.relative_to(self.out)):sha(path)}
        if node['kind']=='online':
            from .engine import result
            warm=self.path(self.lookup[node['parents'][0]])
            calibration=self.out/f"datasets/{node['dataset']}/calibration/rho_{node.get('rho')}.json"
            value=result(node,ck,self.hash,sha(warm),sha(calibration) if node['variant']=='fc' else None)
            dest=self.out/node['id']/'result.json'
            if dest.exists():
                if read_json(dest)!=value:raise ValueError('Result/checkpoint mismatch')
            else:write_json(dest,value)
            trace=self.out/node['id']/'trace.csv.gz';raw=self.trace_bytes(ck['rows'])
            if trace.exists() and receipt.exists():
                if trace.read_bytes()!=raw:raise ValueError('Trace/checkpoint mismatch')
            else:atomic(trace,lambda f:f.write(raw),True)
            artifacts.update({str(p.relative_to(self.out)):sha(p) for p in (dest,trace)})
        record=dict(protocol_hash=self.hash,node_id=node['id'],parents=ck['parents'],artifacts=artifacts)
        if receipt.exists():
            if read_json(receipt)!=record:raise ValueError('Completed lineage mismatch')
        else:write_json(receipt,record)
        self.state['jobs'][node['id']]=dict(status='COMPLETE',artifact=str(path.relative_to(self.out)),sha256=sha(path))
    def commit(self,node,value):
        self.check_frozen_identity()  # All integrity I/O stays outside compute timers.
        value=dict(value,**self.metadata(node),committed_at=now())
        self.validate(node,value)
        if node['kind'] in ('calibration','lambda') and value['complete']:write_json(self.path(node),value)
        else:save_checkpoint(self.partial(node),value)
        if node['kind']=='online':
            raw=self.trace_bytes(value['rows']);atomic(self.out/node['id']/'trace.csv.gz',lambda f:f.write(raw),True)
        if value['complete']:self.finish(node,value)
        self.write_state(**self.projection(node,value))
        return value
    def reconcile(self):
        # Inspect ALL master artifacts before executing the selected subset.
        latest=None
        for node in self.plan['nodes']:
            self.current=node
            old=self.state['jobs'].get(node['id'],{})
            if old.get('status')=='FAILED':raise RuntimeError('FAILED job: '+node['id'])
            ck=self.load(node)
            if ck is None:
                if old.get('status')=='COMPLETE' or self.receipt(node).exists():raise ValueError('Missing completed artifact: '+node['id'])
                if (self.out/node['id']/'result.json').exists():raise ValueError('Orphan result without checkpoint')
                self.state['jobs'][node['id']]=dict(status='PENDING')
                continue
            self.validate(node,ck)
            if latest is None or ck['committed_at']>latest[1]['committed_at']:latest=(node,ck)
            if ck['complete']:self.finish(node,ck)
            else:
                if old.get('status')=='COMPLETE' or self.receipt(node).exists():raise ValueError('Completed job regressed')
                self.state['jobs'][node['id']]=dict(status='INTERRUPTED',artifact=str(self.partial(node).relative_to(self.out)))
            if node['kind']=='calibration':self.state['calibration'][node['dataset']]='COMPLETE' if ck['complete'] else 'INTERRUPTED'
            if node['kind']=='lambda':self.state['frozen_lambdas'][node['dataset']][str(node['rho'])]=ck['lambda_consistency']
        self.current=None
        if latest is not None:
            node,ck=latest
            path=self.path(node) if ck['complete'] else self.partial(node)
            self.state.update(latest_checkpoint=str(path.relative_to(self.out)),last_checkpoint_timestamp=ck['committed_at'])
        completion=self.out/'completion_manifest.json'
        if completion.exists():
            record=read_json(completion)
            if record['protocol_hash']!=self.hash or record['online_trajectories']!=90:raise ValueError('Completion identity mismatch')
            for rel,value in record['artifacts'].items():
                if sha(self.out/rel)!=value:raise ValueError('Completion artifact mismatch: '+rel)
            if any(self.state['jobs'][j['id']]['status']!='COMPLETE' for j in self.plan['nodes']):raise ValueError('Premature master completion')
        self.write_state()
    def run_node(self,node):
        from . import engine, calibration
        saved=self.load(node)
        if saved is not None:self.validate(node,saved)
        if saved is not None and saved['complete']:
            self.finish(node,saved);return
        self.stop();self.current=node
        self.state['jobs'][node['id']]=dict(status='RUNNING')
        if saved is not None:
            self.write_state(**self.projection(node,saved))
        else:
            self.write_state(current_stage=node['kind'].upper(),current_dataset=node['dataset'],current_seed=node['seed'],
                current_rho=node.get('rho'),current_variant=node['variant'],current_warmup_epoch=None,current_warmup_batch=None,
                current_warmup_batch_total=None,current_online_observation=0,current_online_total=None,accepted_updates=0,realized_update_fraction=None)
        spec=self.specs[node['dataset']];data=self.data[node['dataset']]
        commit=lambda value:self.commit(node,value)
        if node['kind']=='warmup':
            coefficient,provenance=self.coefficient(node)
            engine.warmup(node,spec,data,coefficient,provenance,saved,commit,self.stop)
        elif node['kind']=='calibration':
            initial=self.load(self.lookup[node['parents'][0]])
            record=calibration.calibrate(spec['input_dim'],data,initial,saved,commit,self.stop)
            commit(record)
        elif node['kind']=='lambda':commit(calibration.derive(self.load(self.lookup[node['parents'][0]]),node['rho']))
        elif node['kind']=='bootstrap':engine.bootstrap(node,spec,data,self.load(self.lookup[node['parents'][0]]),commit,self.stop)
        elif node['kind']=='online':
            coefficient,provenance=self.coefficient(node)
            initial=self.load(self.lookup[node['parents'][0]])
            boot=self.load(self.lookup[node['parents'][1]])['errors'] if node['variant']=='fc' else []
            engine.online(node,spec,data,initial,boot,coefficient,provenance,saved,commit,self.stop)
        else:raise ValueError('Unknown job kind')
        self.current=None
    def run(self):
        self.preflight();self.reconcile()
        for node in self.selected:
            self.stop()
            if self.state['jobs'][node['id']]['status']=='COMPLETE':continue
            self.run_node(node)
        self.check_frozen_identity()
        verify_datasets()
        if self.state['completed_online_trajectories']==90:
            if not (self.out/'completion_manifest.json').exists():
                self.write_state(current_stage='REPORTING')
                from .reporting import aggregate
                artifacts=aggregate(self.out,self.plan)
                for node in self.plan['nodes']:
                    record=read_json(self.receipt(node));artifacts.update(record['artifacts'])
                    artifacts[str(self.receipt(node).relative_to(self.out))]=sha(self.receipt(node))
                write_json(self.out/'completion_manifest.json',dict(protocol_hash=self.hash,online_trajectories=90,
                    scientific_warmups=90,completed_at=now(),artifacts=artifacts))
            status='COMPLETE';stage='COMPLETE'
        else:status='PENDING';stage='SUBSET_COMPLETE'
        self.write_state(status=status,current_stage=stage,current_dataset=None,current_seed=None,current_rho=None,
            current_variant=None,current_warmup_epoch=None,current_warmup_batch=None,current_warmup_batch_total=None,
            current_online_observation=0,current_online_total=None,accepted_updates=0,realized_update_fraction=None)


def execute(out,filters):
    if not __debug__:raise RuntimeError('Scientific invariant assertions require Python without -O')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'runner.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another orchestrator holds the experiment lock')
        runner=None;handlers={}
        try:
            runner=Runner(out,filters)
            def requested(signum,frame):runner.stop_requested=True
            for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
                handlers[sig]=signal.signal(sig,requested)
            runner.run()
        except (Interrupted,KeyboardInterrupt) as ex:
            if runner is not None:
                if runner.current is not None:
                    key=runner.current['id'];entry=runner.state['jobs'].get(key,{})
                    if entry.get('status')!='COMPLETE':runner.state['jobs'][key]=dict(status='INTERRUPTED')
                    if runner.current['kind']=='calibration' and entry.get('status')!='COMPLETE':
                        runner.state['calibration'][runner.current['dataset']]='INTERRUPTED'
                runner.write_state(status='INTERRUPTED',last_error=dict(kind='external_interruption',message=str(ex),timestamp=now()))
        except Exception as ex:
            if runner is not None:
                if runner.state.get('status')!='FAILED':
                    if runner.current is not None:runner.state['jobs'][runner.current['id']]=dict(status='FAILED',error=repr(ex))
                    if runner.current is not None and runner.current['kind'] in ('calibration','lambda'):
                        runner.state['calibration'][runner.current['dataset']]='FAILED'
                    failure=dict(kind='failure',message=repr(ex),timestamp=now(),
                        node_id=runner.current['id'] if runner.current else None,
                        recovery='Manual investigation required. No automatic failed-job reset or artifact deletion.')
                    marker=runner.out/'failure.json'
                    if marker.exists():failure=read_json(marker)
                    else:write_json(marker,failure)
                    if runner.current is not None:
                        marker=runner.out/(runner.current['id']+'.failure.json')
                        if not marker.exists():write_json(marker,failure)
                    runner.write_state(status='FAILED',last_error=failure)
            raise
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)
