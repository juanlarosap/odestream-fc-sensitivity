"""Top-level frozen-plan CLI. Scientific imports occur only for explicit run."""
import argparse
import json
from pathlib import Path
from .config import DEFAULT_OUTPUT, DATASETS, SEEDS, RHOS, master_plan, selection, ROOT
from .checkpointing import read_json


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for command in ('plan','run'):
        q=sub.add_parser(command)
        q.add_argument('--dataset',choices=DATASETS)
        q.add_argument('--seed',type=int,choices=SEEDS)
        q.add_argument('--rho',type=float,choices=RHOS)
        q.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
        if command=='plan':q.add_argument('--json',action='store_true',help='Print full master plan plus selected IDs')
    return p


def show_plan(args):
    plan=master_plan();selected=selection(plan,args.dataset,args.seed,args.rho)
    selected_ids={j['id'] for j in selected}
    path=args.output/'state.json';state=read_json(path) if path.exists() else {}
    if state and state['protocol_hash']!=plan['protocol_hash']:raise ValueError('Existing state protocol mismatch')
    jobs=state.get('jobs',{})
    if args.json:
        print(json.dumps(dict(master=plan,selected_ids=[j['id'] for j in selected],existing_jobs=jobs),indent=2));return
    print('FC sensitivity: fixed master plan')
    print('Protocol hash:',plan['protocol_hash'])
    print('Dataset order:',', '.join(DATASETS));print('Seed order:',SEEDS);print('FC rho order:',RHOS)
    print('Master: 15 Control + 75 FC = 90 online; 90 distinct scientific warmups.')
    print('Preparation: 3 separate Control seed-0 warmups → 3 calibration references → 15 rho-specific lambdas.')
    print('Calibration/bootstrap nodes and full warmup dependencies:')
    for node in plan['nodes']:
        state_name=jobs.get(node['id'],{}).get('status','PENDING')
        print(f"{'SELECTED' if node['id'] in selected_ids else '        '} [{state_name}] {node['id']} <- {', '.join(node['parents']) or '(fresh seeded initialization)'}")
    warm=sum(j['kind']=='warmup' and j['scientific'] for j in selected)
    online=[j for j in selected if j['kind']=='online']
    complete=sum(jobs.get(j['id'],{}).get('status')=='COMPLETE' for j in plan['online_jobs'])
    print(f'Selected execution subset: {len(online)} online, {warm} scientific warmups; master remains 90 online.')
    print(f'Master recorded online completion: {complete}/90; remaining {90-complete}.')
    print('Existing status is a read-only projection; run validates artifacts before trusting completion.')


def main(argv=None):
    args=parser().parse_args(argv)
    if args.command=='plan':return show_plan(args)
    # Keep artifacts outside source/historical trees even with --output overrides.
    resolved=args.output.resolve()
    if resolved==ROOT or resolved.is_relative_to(ROOT/'experiments') or resolved.is_relative_to(ROOT/'data'):
        raise ValueError('Output must be a dedicated artifacts directory outside experiments/ and data/')
    from .runner import execute
    execute(resolved,dict(dataset=args.dataset,seed=args.seed,rho=args.rho))


if __name__=='__main__':main()
