"""Paired descriptive results; reporting performs no model work."""
import csv
import io
import math
from .checkpointing import write_json, atomic, sha, read_json
from .config import DATASETS, SEEDS, RHOS


def summary(values):
    if len(values)!=5 or not all(math.isfinite(v) for v in values):raise ValueError('Expected five finite seeds')
    import numpy as np
    a=np.asarray(values,dtype=float)
    return dict(mean=float(a.mean()),sample_sd=float(a.std(ddof=1)),min=float(a.min()),max=float(a.max()),seeds=a.tolist())


def aggregate(out,plan):
    rows=[read_json(out/j['id']/'result.json') for j in plan['online_jobs']]
    lookup={(r['dataset'],r['seed'],r['variant'],r.get('rho_target')):r for r in rows}
    groups={}
    for d in DATASETS:
        controls=[lookup[d,s,'control',None] for s in SEEDS]
        groups[d]=dict(control={k:summary([r[k] for r in controls]) for k in ('MSE','MAE','online_compute_wall_sec')},fc={})
        for rho in RHOS:
            fc=[lookup[d,s,'fc',rho] for s in SEEDS]
            if any(a['MSE']<=0 or a['online_compute_wall_sec']<=0 for a in controls):raise ValueError('Undefined paired ratio')
            groups[d]['fc'][str(rho)]=dict(metrics={k:summary([r[k] for r in fc]) for k in
                ('MSE','MAE','online_compute_wall_sec','updates','eligible_online_steps','update_fraction')},
                relative_MSE_change=summary([b['MSE']/a['MSE']-1 for a,b in zip(controls,fc)]),
                runtime_saving=summary([1-b['online_compute_wall_sec']/a['online_compute_wall_sec'] for a,b in zip(controls,fc)]))
    report=out/'reports'
    write_json(report/'aggregate.json',groups)
    fields=['dataset','seed','variant','rho_target','MSE','MAE','online_compute_wall_sec','updates','eligible_online_steps','update_fraction']
    buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=fields);writer.writeheader()
    writer.writerows({k:r.get(k) for k in fields} for r in rows)
    atomic(report/'summary.csv',lambda f:f.write(buf.getvalue()))
    text='# FC sensitivity\n\nFive paired seeds. Ratios are computed per seed before aggregation; sample SD uses ddof=1. No rho is selected. MSE/MAE use TRAIN-standardized targets. Runtime is the frozen online compute interval.\n\n'
    text+='| Dataset | Rho | FC MSE mean | Relative MSE change mean | Runtime saving mean |\n|---|---:|---:|---:|---:|\n'
    for d in DATASETS:
        for rho in RHOS:
            g=groups[d]['fc'][str(rho)]
            text+=f"| {d} | {rho} | {g['metrics']['MSE']['mean']:.9g} | {g['relative_MSE_change']['mean']:.9g} | {g['runtime_saving']['mean']:.9g} |\n"
    atomic(report/'REPORT.md',lambda f:f.write(text))
    return {str(p.relative_to(out)):sha(p) for p in (report/'aggregate.json',report/'summary.csv',report/'REPORT.md')}
