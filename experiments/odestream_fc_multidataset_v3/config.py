"""Scientific identity and the complete, filter-independent master dependency plan."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ('ETTh2', 'ECL', 'Weather')
SEEDS = (0, 1, 2, 3, 4)
RHOS = (0.5, 0.75, 1.0, 1.25, 1.5)
EXPERIMENT_ID = 'odestream-fc-sensitivity-v3'
DEFAULT_OUTPUT = ROOT / 'outputs' / EXPERIMENT_ID
WEATHER_FEATURES = ['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
    'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)',
    'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)', 'rain (mm)',
    'raining (s)', 'SWDR (W/m\ufffd)', 'PAR (\ufffdmol/m\ufffd/s)',
    'max. PAR (\ufffdmol/m\ufffd/s)', 'Tlog (degC)', 'OT']
ETT_FEATURES = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
ANOMALIES = [dict(previous_row=19043, row=19044, previous='2020-05-12 06:00:00',
    current='2020-05-12 06:00:00', seconds=0, kind='duplicate timestamp and entire row'),
    dict(previous_row=21513, row=21514, previous='2020-05-29 09:30:00',
    current='2020-05-29 11:10:00', seconds=6000, kind='gap of nine missing timestamps')]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def registry():
    specs = [
        ('ETTh2', 17420, 3484, 4355, ETT_FEATURES, ETT_FEATURES, 'OT',
         'a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b'),
        ('ECL', 26304, 5260, 6576, [f'series_{i:03d}' for i in range(1,322)], ['series_321'], 'series_321',
         '583f9538d489bfd29ba5edb3f3c2b51152ad8900468a8bbc17e1f0a6f2118ffd'),
        ('Weather', 52696, 10539, 13174, WEATHER_FEATURES, WEATHER_FEATURES, 'OT',
         '34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63')]
    return {name: dict(file=f'data/{name}.csv', sha256=sha, rows=n,
        numerical_columns=columns, inputs=inputs, target=target, input_dim=len(inputs),
        task='univariate_to_univariate' if name=='ECL' else 'multivariate_to_univariate',
        role='development_mechanistic' if name=='ETTh2' else 'confirmatory',
        train=[0,train], validation=[train,offline], online=[offline,n],
        train_windows=train-24, validation_windows=offline-train, online_windows=n-offline,
        bootstrap_targets=[offline-256,offline], scaler_fit_rows=[0,train],
        cadence_seconds=600 if name=='Weather' else 3600,
        timestamp_anomalies=ANOMALIES if name=='Weather' else [],
        timestamp_provenance='user-specified hourly wrapper over electricity.txt.gz' if name=='ECL' else 'file timestamps',
        row_policy='preserve exact recorded rows, order and values')
        for name,n,train,offline,columns,inputs,target,sha in specs}


def protocol():
    return dict(version=3, experiment_id=EXPERIMENT_ID, datasets=registry(),
        dataset_order=list(DATASETS), seeds=list(SEEDS), rho_targets=list(RHOS),
        condition_order=['control/always']+[f'fc/rho_{r}/r50' for r in RHOS],
        lag=24, horizon=1, time_grid='float32 numpy.linspace(0,24,num=24); recorded observations',
        scaler='StandardScaler fit on selected TRAIN rows only; float32 windows',
        model=dict(hidden_dim=64,latent_dim=64,gru_layers=1,lstm_layers=2,output_dim=1,
                   fusion='Linear(2,1)',ode='historical Euler/custom adjoint',ode_h_max=.05,time_invariant=True),
        stochasticity='one randn_like(mu) per student forward; stochastic validation; exact shared teacher epsilon',
        loss=dict(name='PaperLoss-B',mse=1.,latent_summed_kl=1.,latent_summed_l1=.01),
        optimizer=dict(name='Adam',lr=.001,betas=[.9,.999],eps=1e-8,weight_decay=0),
        warmup=dict(batch_size=64,drop_last=True,shuffle=False,max_epochs=100,patience=10,
                    validation='all windows, sample-weighted stochastic student MSE; strict improvement',
                    batch_rng='historical NumPy multinomial even with one valid temporal start',
                    transition='best VAE weights; final validation RNG; new LSTM/fusion; fresh Adam; new exact-copy teacher'),
        fc=dict(beta=.99,teacher_stop_gradient=True,loss='mean squared prediction difference',
                phases=['warmup','online'],ordering='student; same-epsilon teacher; backward; Adam; EMA',
                skip='student forward and RNG retained; no teacher/Adam/EMA; append error'),
        r50=dict(window=256,quantile=.5,method='linear',ties='error >= threshold',
                 error='current pre-update squared standardized target error',label_delay=0,
                 threshold_history='prior FIFO only; append current error after decision/update including skips',
                 bootstrap='last 256 validation targets, stochastic frozen student; restore initial RNG',forced_rate=False),
        calibration=dict(seed=0,steps=128,batch_size=1,lambda_zero=True,target_rows=[24,152],
            input_row_union=[0,151],eps=1e-12,exclude='G_cons == 0 only',
            reference='median G_cons/(G_base+eps) over positive G_cons',lambda_formula='rho_target / dataset_reference',
            initialization='separate preparation Control warmup; validation-selected; fresh Adam and checkpoint RNG',
            q95_policy='report each rho; diagnostic only, no selection or numerical cap'),
        execution=dict(device='cpu',intra_threads=8,inter_threads=8,deterministic=True),
        checkpoint=dict(warmup_batches=10,online_observations=250,epoch_boundaries=True),
        timing=dict(name='online_compute_wall_sec',
            included='batch retrieval and complete online_step including finite checks and built-in diagnostics',
            excluded='data preparation,warmup,calibration,bootstrap,initialization,restoration,checkpoint,trace,progress,log,report I/O'),
        reporting='paired same-seed FC/Control MSE minus one; 1-FC/Control runtime; mean, sample SD ddof=1, min,max,seeds',
        scientific_warmups=90,scientific_online_trajectories=90)


def master_plan():
    nodes=[]; online=[]
    def add(identifier,kind,dataset,parents=(),seed=0,rho=None,variant='control',scientific=False):
        node=dict(id=identifier,kind=kind,dataset=dataset,seed=seed,variant=variant,
                  parents=list(parents),scientific=scientific)
        if rho is not None: node['rho']=rho
        nodes.append(node)
        return node
    for d in DATASETS:
        prefix=f'datasets/{d}'
        prep=f'{prefix}/preparation/control_seed_0'
        ref=f'{prefix}/calibration/reference'
        add(prep,'warmup',d)
        add(ref,'calibration',d,[prep])
        for rho in RHOS:add(f'{prefix}/calibration/rho_{rho}','lambda',d,[ref],rho=rho,variant='fc')
        for seed in SEEDS:
            control=f'{prefix}/control/seed_{seed}'
            warm=control+'/warmup'
            add(warm,'warmup',d,[ref],seed=seed,scientific=True)
            online.append(add(control+'/always','online',d,[warm],seed=seed,scientific=True))
            for rho in RHOS:
                path=f'{prefix}/fc/rho_{rho}/seed_{seed}'
                lam=f'{prefix}/calibration/rho_{rho}'
                add(path+'/warmup','warmup',d,[lam],seed,rho,'fc',True)
                add(path+'/bootstrap','bootstrap',d,[path+'/warmup'],seed,rho,'fc')
                online.append(add(path+'/r50','online',d,[path+'/warmup',path+'/bootstrap'],seed,rho,'fc',True))
    return dict(protocol_hash=digest(protocol()),nodes=nodes,online_jobs=online,
                counts=dict(datasets=3,seeds=5,rhos=5,control_online=15,fc_online=75,online=90,scientific_warmups=90))


def selection(plan,dataset=None,seed=None,rho=None):
    if dataset is not None and dataset not in DATASETS:raise ValueError('Unknown dataset')
    if seed is not None and seed not in SEEDS:raise ValueError('Unknown seed')
    if rho is not None and rho not in RHOS:raise ValueError('Unknown rho')
    selected=[j for j in plan['online_jobs'] if (dataset is None or j['dataset']==dataset)
        and (seed is None or j['seed']==seed) and (rho is None or j['variant']=='control' or j['rho']==rho)]
    lookup={j['id']:j for j in plan['nodes']}; wanted=set()
    def visit(key):
        if key in wanted:return
        wanted.add(key)
        for parent in lookup[key]['parents']:visit(parent)
    for j in selected:
        visit(j['id'])
        # Preparation/calibration precedes scientific Control too.
        visit(f"datasets/{j['dataset']}/calibration/reference")
    return [j for j in plan['nodes'] if j['id'] in wanted]


def artifact_path(out,node):
    if node['kind'] in ('calibration','lambda'):
        return Path(out)/(node['id']+'.json')
    return Path(out)/node['id']/'state.pt'
