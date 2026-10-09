"""Public condition identities and a filter-independent 30-trajectory plan."""
import copy
from pathlib import Path
from experiments.odestream_fc_multidataset_v3 import config as v3

ROOT = v3.ROOT
DATASETS, SEEDS = v3.DATASETS, v3.SEEDS
EXPERIMENT_ID = 'odestream-fc-ablation-v1'
DEFAULT_OUTPUT = ROOT / 'outputs' / EXPERIMENT_ID
DEFAULT_V3_OUTPUT = v3.DEFAULT_OUTPUT
CONDITIONS = {
    'r50_only': dict(fc_enabled=False, gate_mode='r50'),
    'fc_only': dict(fc_enabled=True, gate_mode='always'),
}


def condition(name):
    if name not in CONDITIONS:
        raise ValueError('Unknown ablation condition: ' + str(name))
    return dict(CONDITIONS[name])


def internal_node(node):
    """Only at v3 API boundaries: variant selects the objective, never the gate."""
    expected = condition(node['condition'])
    if any(node.get(k) != v for k, v in expected.items()):
        raise ValueError('Condition dispatch mismatch')
    if node.get('rho_target') != (1.0 if expected['fc_enabled'] else None):
        raise ValueError('Only FC rho_target=1.0 is permitted')
    value = dict(node, variant='fc' if expected['fc_enabled'] else 'control')
    if expected['fc_enabled']:
        value['rho'] = 1.0
    return value


def protocol():
    value = copy.deepcopy(v3.protocol())
    value.update(experiment_id=EXPERIMENT_ID, version=1,
                 parent_protocol_hash=v3.digest(v3.protocol()),
                 conditions=copy.deepcopy(CONDITIONS),
                 condition_order=list(CONDITIONS), rho_targets=[1.0],
                 scientific_warmups=30, scientific_online_trajectories=30,
                 calibration_policy='Read-only authoritative v3 rho_1.0 artifacts; never calibrate',
                 interpretation='FC treatment includes independent FC-aware warmup and online FC',
                 reporting='Standalone conditions; five seeds; mean, sample SD ddof=1, min, max, seeds')
    return value


def master_plan():
    nodes = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for name in CONDITIONS:
                base = f'datasets/{dataset}/{name}/seed_{seed}'
                common = dict(dataset=dataset, seed=seed, condition=name, **condition(name))
                if common['fc_enabled']:
                    common['rho_target'] = 1.0
                warm = dict(common, id=base+'/warmup', kind='warmup', parents=[], scientific=True)
                nodes.append(warm)
                parents = [warm['id']]
                if common['gate_mode'] == 'r50':
                    boot = dict(common, id=base+'/bootstrap', kind='bootstrap',
                                parents=[warm['id']], scientific=False)
                    nodes.append(boot)
                    parents.append(boot['id'])
                nodes.append(dict(common, id=base+'/online', kind='online',
                                  parents=parents, scientific=True))
    return dict(protocol_hash=v3.digest(protocol()), nodes=nodes,
                online_jobs=[n for n in nodes if n['kind'] == 'online'],
                counts=dict(online=30, warmups=30, bootstraps=15, calibration=0))


def selection(plan, dataset=None, seed=None, conditions=None):
    if dataset is not None and dataset not in DATASETS:
        raise ValueError('Unknown dataset')
    if seed is not None and seed not in SEEDS:
        raise ValueError('Unknown scientific seed')
    names = list(CONDITIONS) if conditions is None else list(conditions)
    if not names or len(set(names)) != len(names):
        raise ValueError('Select distinct conditions')
    for name in names:
        condition(name)
    return [n for n in plan['nodes'] if n['condition'] in names
            and (dataset is None or n['dataset'] == dataset)
            and (seed is None or n['seed'] == seed)]


def safe_output(path, v3_output=DEFAULT_V3_OUTPUT):
    out, old = Path(path).resolve(), Path(v3_output).resolve()
    forbidden = [ROOT, ROOT/'experiments', ROOT/'data', ROOT/'.git', DEFAULT_V3_OUTPUT, old]
    for protected in forbidden:
        if out == protected or protected.is_relative_to(out):
            raise ValueError('Output overlaps protected directory: ' + str(protected))
        if protected != ROOT and out.is_relative_to(protected):
            raise ValueError('Output is inside protected directory: ' + str(protected))
    if out.exists():
        if any(p.is_symlink() for p in out.rglob('*')):
            raise ValueError('Output contains symlinked artifacts')
        from experiments.odestream_fc_multidataset_v3.checkpointing import read_json
        identity = out/'protocol.json'
        if identity.exists():
            if read_json(identity).get('experiment_id') != EXPERIMENT_ID:
                raise ValueError('Output belongs to another experiment')
        elif any(out.iterdir()):
            raise ValueError('Nonempty output without ablation protocol')
    return out
