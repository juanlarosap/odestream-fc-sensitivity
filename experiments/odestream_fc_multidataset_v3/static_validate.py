"""Read-only syntax/plan/CLI checks. Never imports or executes the scientific engine."""
import ast
import contextlib
from datetime import datetime
import io
from pathlib import Path
import sys
from .config import DATASETS, SEEDS, RHOS, master_plan, selection, registry, protocol, digest
from .checkpointing import now
from .status import percentages
from .run import parser, main


def validate():
    folder=Path(__file__).parent
    for path in folder.glob('*.py'):compile(path.read_text(),str(path),'exec')
    plan=master_plan();nodes=plan['nodes'];online=plan['online_jobs']
    assert DATASETS==('ETTh2','ECL','Weather') and SEEDS==tuple(range(5)) and RHOS==(.5,.75,1.,1.25,1.5)
    assert len(nodes)==276 and len({j['id'] for j in nodes})==276
    assert len(online)==90
    assert sum(j['variant']=='control' for j in online)==15
    assert sum(j['variant']=='fc' for j in online)==75
    warm=[j for j in nodes if j['kind']=='warmup' and j['scientific']]
    prep=[j for j in nodes if j['kind']=='warmup' and not j['scientific']]
    assert len(warm)==90 and len(prep)==3
    assert not ({j['id'] for j in warm}&{j['id'] for j in prep})
    assert all('rho' not in j for j in nodes if j['variant']=='control')
    assert len({(j['dataset'],j['rho'],j['seed']) for j in warm if j['variant']=='fc'})==75
    seen=set()
    for node in nodes:
        assert set(node['parents'])<=seen
        seen.add(node['id'])
    pilot=selection(plan,'ETTh2',0,1.)
    assert [j['kind'] for j in pilot]==['warmup','calibration','lambda','warmup','online','warmup','bootstrap','online']
    po=[j for j in pilot if j['kind']=='online']
    assert len(po)==2 and {j['variant'] for j in po}=={'control','fc'}
    assert sum(j['kind']=='warmup' and j['scientific'] for j in pilot)==2
    assert plan==master_plan() and plan['protocol_hash']==digest(protocol())
    # A completed pilot belongs to the same unfiltered job IDs; no scenario/model execution.
    assert {j['id'] for j in po}<={j['id'] for j in online}
    assert len([j for j in online if j['id'] not in {x['id'] for x in po}])==88
    assert [registry()[d]['input_dim'] for d in DATASETS]==[7,1,21]
    assert [registry()[d]['online_windows'] for d in DATASETS]==[13065,19728,39522]
    assert registry()['ECL']['inputs']==['series_321']
    assert protocol()['execution']==dict(device='cpu',intra_threads=8,inter_threads=8,deterministic=True)
    for n in (0,2,43,90):assert percentages(n)==(100*n/90,100*(90-n)/90)
    assert datetime.fromisoformat(now()).utcoffset() is not None
    parsed=parser().parse_args(['run','--dataset','ETTh2','--seed','0','--rho','1.0'])
    assert parsed.rho==1.0  # Parse only; NEVER dispatch run.
    for argv in (['plan'],['plan','--dataset','ETTh2','--seed','0','--rho','1.0']):
        with contextlib.redirect_stdout(io.StringIO()) as capture:main(argv)
        assert 'master remains 90' in capture.getvalue()
    # Status's dependency closure is deliberately small and contains no scientific imports.
    status=ast.parse((folder/'status.py').read_text())
    forbidden={'write_json','atomic','save_checkpoint','execute','warmup','calibrate','online','Model'}
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in forbidden for n in ast.walk(status))
    for name in ('torch','numpy','pandas'):
        assert name not in sys.modules, f'Static validation imported scientific library {name}'
    return dict(syntax='PASS',plan='PASS',pilot_selection='PASS',cli='PASS',
                master_online=90,scientific_warmups=90,pilot_online=2,pilot_scientific_warmups=2,
                scientific_execution=False)


if __name__=='__main__':
    import json
    print(json.dumps(validate(),indent=2))
