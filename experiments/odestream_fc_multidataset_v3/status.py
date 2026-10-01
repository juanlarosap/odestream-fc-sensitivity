"""Strictly read-only, standard-library-only committed-progress dashboard."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import socket
from .config import DEFAULT_OUTPUT, digest, protocol, EXPERIMENT_ID
from .checkpointing import now, read_json


def percentages(completed,total=90):
    if not 0<=completed<=total:raise ValueError('Invalid completion count')
    return 100*completed/total,100*(total-completed)/total


def snapshot(out=DEFAULT_OUTPUT):
    path=Path(out)/'state.json'
    state=read_json(path) if path.exists() else dict(experiment_id=EXPERIMENT_ID,protocol_hash=digest(protocol()),
        status='PENDING',current_stage='NOT_STARTED',completed_warmups=0,completed_online_trajectories=0)
    state=dict(state,status_timestamp=now(),process_note='No active-process evidence')
    failure=Path(out)/'failure.json'
    if failure.exists():state.update(status='FAILED',last_error=read_json(failure))
    if state['status']=='RUNNING':
        local=state.get('hostname')==socket.gethostname()
        if local and state.get('pid'):
            try:os.kill(state['pid'],0)
            except ProcessLookupError:
                state.update(status='INTERRUPTED',process_note='Recorded local process is absent; checkpoint validation occurs on run')
            except PermissionError:state['process_note']='Recorded process exists; permission restricted'
            else:state['process_note']='Recorded PID exists locally; PID reuse is possible'
        else:state['process_note']='Remote process activity unknown; SSH/VPN state does not imply scientific failure'
        stamp=state.get('last_state_update_timestamp')
        if stamp and (datetime.fromisoformat(state['status_timestamp'])-datetime.fromisoformat(stamp)).total_seconds()>3600:
            state['process_note']+='; state over one hour old, potentially stale (long compute intervals are possible)'
    return state


def dashboard(state):
    get=lambda key:state.get(key) if state.get(key) is not None else '—'
    lines=['Experiment: FC sensitivity',f"Status timestamp: {state['status_timestamp']}",
        f"State: {get('status')}",f"Protocol hash: {get('protocol_hash')}",
        f"Current dataset: {get('current_dataset')}",f"Current seed: {get('current_seed')}",
        f"Current rho: {get('current_rho')}",f"Current variant: {get('current_variant')}",
        f"Current stage: {get('current_stage')}",'Preparation:']
    for d in protocol()['dataset_order']:
        lines.append(f"  {d}: {state.get('calibration',{}).get(d,'PENDING')}; frozen lambdas: {state.get('frozen_lambdas',{}).get(d,{})}")
    for title,key in [('Scientific warmups','completed_warmups'),('Online trajectories','completed_online_trajectories')]:
        n=state.get(key,0);done,remaining=percentages(n)
        lines.append(f'{title}: {n}/90; {done:.2f}% complete; {remaining:.2f}% remaining')
    if state.get('current_stage')=='WARMUP' and state.get('current_warmup_batch_total'):
        b=state.get('current_warmup_batch',0);total=state['current_warmup_batch_total']
        lines.append(f"Current warmup: {get('current_warmup_epoch')} completed epochs; committed batch {b}/{total} ({100*b/total:.2f}% of current epoch)")
        lines.append('Warmup total duration/progress unknown until early stopping; epoch cap 100.')
    if state.get('current_stage')=='ONLINE' and state.get('current_online_total'):
        n=state.get('current_online_observation',0);total=state['current_online_total']
        lines.append(f'Current online: {n}/{total} ({100*n/total:.2f}%)')
        if state.get('current_variant')=='fc':lines.append(f"R50 accepted updates: {get('accepted_updates')}; realized update fraction: {get('realized_update_fraction')}")
    lines += [f"Current/last checkpoint: {get('latest_checkpoint')}",
        f"Last successful checkpoint: {get('last_checkpoint_timestamp')}",
        f"Last state update: {get('last_state_update_timestamp')}",f"Last error: {state.get('last_error') or 'none'}",
        f"Process: {state.get('process_note','unknown')}",'ETA: unavailable',
        'Progress reflects committed checkpoints; completion percentages count jobs, not runtime.']
    return '\n'.join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args(argv)
    print(dashboard(snapshot(args.output)))


if __name__=='__main__':main()
