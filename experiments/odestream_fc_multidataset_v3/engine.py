"""Single scientific engine. Functions are called only by the explicit run command."""
import copy
import time
import numpy as np
import torch
from . import runtime
from .datasets import batch
from .losses import paperloss, finite, equal, diagnostics, optimizer
from .model_adapter import Model, combined, online_boundary
from .r50 import Gate
from .timing import timed_observation

LOSS_KEYS=('mse_loss','kl_loss','l1_loss','total_loss')


def validate_state(state,variant,updates=None):
    finite(state['model'],'checkpoint model');finite(state['optimizer'],'checkpoint Adam')
    if not state['model'] or any(v.device.type!='cpu' for v in state['model'].values()):raise ValueError('Invalid CPU model state')
    groups=state['optimizer']['param_groups']
    if len(groups)!=1:raise ValueError('Unexpected optimizer groups')
    g=groups[0]
    if (g['lr'],tuple(g['betas']),g['eps'],g['weight_decay'])!=(.001,(.9,.999),1e-8,0):raise ValueError('Adam contract mismatch')
    for item in state['optimizer']['state'].values():
        if updates is not None and float(item['step'])!=updates:raise ValueError('Adam committed count mismatch')
    if updates==0 and state['optimizer']['state']:raise ValueError('Expected fresh Adam')
    if updates and not state['optimizer']['state']:raise ValueError('Missing Adam state')
    if updates and set(state['optimizer']['state'])!=set(g['params']):raise ValueError('Incomplete Adam parameter state')
    method=state['method']
    if variant=='fc':
        if method is None or method['beta']!=.99 or method['lambda_consistency']<=0:raise ValueError('Invalid FC state')
        finite(method,'FC state')
        if not equal(method['student'],state['model']):raise ValueError('Student state mismatch')
        count=method['student_update_count']
        if method['teacher_update_count']!=count or len(method['rows'])!=count:raise ValueError('Teacher counter mismatch')
        if updates is not None and count!=updates:raise ValueError('FC committed count mismatch')
        if method['teacher'].keys()!=method['student'].keys():raise ValueError('Teacher schema mismatch')
    elif method is not None:raise ValueError('Control has FC state')
    import random
    random.Random().setstate(state['rng']['python'])
    np.random.RandomState().set_state(state['rng']['numpy'])
    r=state['rng']['torch']
    if not isinstance(r,torch.Tensor) or r.dtype!=torch.uint8 or r.device.type!='cpu' or r.ndim!=1:raise ValueError('Invalid Torch RNG')


def validate_warmup(ck,node,spec):
    batches=spec['train_windows']//64
    if ck['complete']:
        if ck['epoch']!=100 and ck['stopping']['bad_epochs']<10:raise ValueError('Premature warmup completion')
        validate_state(ck['state'],node['variant'],0)
        validate_state(ck['warmup_final_state'],node['variant'],ck['epoch']*batches)
        if not ck['history'] or len(ck['history'])!=ck['epoch']:raise ValueError('Warmup history mismatch')
        best=ck['best']['model']
        embedded={k.removeprefix('model1.'):v for k,v in ck['state']['model'].items() if k.startswith('model1.')}
        if not equal(best,embedded):raise ValueError('Best VAE transition mismatch')
        if node['variant']=='fc' and not equal(ck['state']['method']['teacher'],ck['state']['model']):raise ValueError('Online teacher is not exact-copy')
    else:
        epoch=ck['epoch'];idx=ck['batch_index']
        if not (0<=epoch<=100 and 0<=idx<=batches):raise ValueError('Warmup position mismatch')
        if ck['count']!=idx*64 or len(ck['history'])!=epoch:raise ValueError('Warmup accumulated count mismatch')
        validate_state(ck['state'],node['variant'],epoch*batches+idx)
        finite(ck['totals'],'warmup totals')
    stopping=ck['stopping']
    if stopping['patience']!=10 or stopping['best_epoch']>ck['epoch'] or stopping['bad_epochs']<0:raise ValueError('Early stopping mismatch')
    if ck['epoch']:
        expected=min(ck['history'],key=lambda r:r['validation_mse'])
        if stopping['best']!=expected['validation_mse'] or stopping['best_epoch']!=expected['epoch']:raise ValueError('Best epoch mismatch')
        if stopping['bad_epochs']!=ck['epoch']-stopping['best_epoch']:raise ValueError('Patience counter mismatch')
        validate_state(ck['best'],node['variant'],stopping['best_epoch']*batches)
    finite(ck['history'],'validation history');finite(ck['warmup_seconds'],'warmup time')


def validation(model,data):
    x,y,t=data;total=count=0
    model.eval()
    with torch.no_grad():
        for start in range(0,len(x),64):
            pred,z,mu,lv=model(x[start:start+64].transpose(0,1),t[start:start+64].transpose(0,1),None)
            finite(pred,'validation prediction');diagnostics(z,mu,lv)
            total+=(pred-y[start:start+64]).square().sum().item();count+=y[start:start+64].numel()
    model.train();finite(total,'validation MSE')
    return total/count


def warmup(node,spec,data,coefficient,provenance,saved,commit,stop):
    runtime.seed(node['seed'])
    model=Model(spec['input_dim'],node['variant'],'warmup',coefficient,provenance)
    epoch=idx=0;history=[];best=None;wall=0.;totals={k:0. for k in LOSS_KEYS};count=0
    stopping=dict(patience=10,best=float('inf'),bad_epochs=0,best_epoch=0)
    if saved is not None:
        model.restore(saved['state']);epoch=saved['epoch'];idx=saved['batch_index'];history=saved['history']
        best=saved['best'];wall=saved['warmup_seconds'];totals=saved['totals'];count=saved['count'];stopping=saved['stopping']
    batches=len(data['train'][0])//64
    def checkpoint():
        return commit(dict(state=model.snapshot(),epoch=epoch,batch_index=idx,history=history,best=best,
            warmup_seconds=wall,totals=totals,count=count,stopping=stopping,complete=False))
    checkpoint()
    while epoch<100 and stopping['bad_epochs']<10:
        while idx<batches:
            stop(checkpoint)
            start=time.perf_counter()
            current=batch(data['train'],idx,64);row=model.update(current)
            for k in totals:totals[k]+=row[k]*len(current[2])
            count+=len(current[2]);idx+=1;wall+=time.perf_counter()-start
            if idx%10==0:checkpoint()
        start=time.perf_counter();val=validation(model.model,data['validation'])
        epoch+=1
        if val<stopping['best']:
            stopping.update(best=val,bad_epochs=0,best_epoch=epoch);best=model.snapshot()
        else:stopping['bad_epochs']+=1
        history.append(dict(epoch=epoch,validation_mse=val,**{k:v/count for k,v in totals.items()}))
        wall+=time.perf_counter()-start
        totals={k:0. for k in LOSS_KEYS};count=0;idx=0
        checkpoint();stop()
    current=model.snapshot();model.model.load_state_dict(best['model'])
    return commit(dict(state=online_boundary(model),epoch=epoch,batch_index=0,history=history,best=best,
        stopping=stopping,warmup_final_state=current,warmup_seconds=wall,complete=True))


def bootstrap(node,spec,data,initial,commit,stop):
    # Student-only frozen bootstrap; no teacher updates or Adam updates.
    model=combined(spec['input_dim']);model.load_state_dict(initial['state']['model'])
    opt=optimizer(model);opt.load_state_dict(copy.deepcopy(initial['state']['optimizer']))
    runtime.restore_rng(initial['state']['rng'])
    before=copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=runtime.rng()))
    rows=[]
    try:
        with torch.no_grad():
            x,y,t=data['validation']
            for i in range(len(x)-256,len(x)):
                stop()
                output=model(x[i:i+1].transpose(0,1),t[i:i+1].transpose(0,1),None)
                pred=output[0].item();diagnostics(*output[-3:]);target=y[i].item()
                error=(target-pred)**2;finite(error,'bootstrap')
                rows.append(dict(dataset_row=spec['validation'][0]+i,target=target,prediction=pred,squared_error=error))
    finally:runtime.restore_rng(before['rng'])
    assert equal(before,dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=runtime.rng()))
    return commit(dict(rows=rows,errors=[r['squared_error'] for r in rows],complete=True))


def online_step(model,gate,current,index,updates):
    model.opt.zero_grad();tau=gate.threshold()
    output,forward=model.forward(current)
    prediction=output[0].item();target=current[2].item();error=(target-prediction)**2
    comp=paperloss(current[2],output[0],*output[-3:]);finite(comp,'prediction losses');finite(error,'error')
    flag=gate.decide(error,tau)
    diag=model.update(current,output,forward) if flag else {}
    gate.append(error)
    return dict(online_index=index,target=target,prediction=prediction,squared_error=error,
        absolute_error=abs(target-prediction),update_flag=flag,cumulative_update_count=updates+flag,
        gate_threshold=tau,rolling_error_statistic=error,diagnostics=diag)


def validate_online(ck,node,spec,boot,data=None):
    total=spec['online_windows'];mode='r50' if node['variant']=='fc' else 'always'
    if ck['n']!=total or ck['next_index']!=len(ck['rows']) or not 0<=ck['next_index']<=total:raise ValueError('Online position mismatch')
    gate=Gate(mode,boot);updates=0
    for i,row in enumerate(ck['rows']):
        error=(row['target']-row['prediction'])**2;tau=gate.threshold();flag=gate.decide(error,tau)
        gate.append(error);updates+=flag
        if (row['online_index']!=i or row['squared_error']!=error or row['absolute_error']!=abs(row['target']-row['prediction'])
            or row['gate_threshold']!=tau or row['update_flag']!=flag or row['cumulative_update_count']!=updates
            or row['rolling_error_statistic']!=error):raise ValueError('Online trace/gate mismatch')
        if data is not None and row['target']!=data['stream'][1][i].item():raise ValueError('Trace target mismatch')
        if not flag and row['diagnostics']:raise ValueError('Skipped update diagnostics')
    if ck['updates']!=updates or ck['window']!=gate.state():raise ValueError('FIFO/update mismatch')
    if ck['sum_squared_error']!=sum(r['squared_error'] for r in ck['rows']):raise ValueError('MSE accumulator mismatch')
    if ck['sum_absolute_error']!=sum(r['absolute_error'] for r in ck['rows']):raise ValueError('MAE accumulator mismatch')
    if ck['complete'] and ck['next_index']!=total:raise ValueError('Premature completion')
    validate_state(ck['state'],node['variant'],updates);finite(ck['rows'],'trace');finite(ck['online_compute_wall_sec'],'online time')
    if ck['online_compute_wall_sec']<0:raise ValueError('Negative time')
    if node['variant']=='fc':
        if not equal(ck['state']['method']['rows'],[r['diagnostics'] for r in ck['rows'] if r['update_flag']]):raise ValueError('FC diagnostic history mismatch')


def online(node,spec,data,initial,boot,coefficient,provenance,saved,commit,stop):
    model=Model(spec['input_dim'],node['variant'],'online',coefficient,provenance,
                saved['state'] if saved is not None else initial['state'])
    rows=[] if saved is None else saved['rows'];wall=0. if saved is None else saved['online_compute_wall_sec']
    updates=0 if saved is None else saved['updates']
    mode='r50' if node['variant']=='fc' else 'always'
    gate=Gate(mode,boot if saved is None else (saved['window'] or []))
    sums=[sum(r['squared_error'] for r in rows),sum(r['absolute_error'] for r in rows)]
    def checkpoint(complete=False):
        return commit(dict(state=model.snapshot(),rows=rows,window=gate.state(),next_index=len(rows),
            n=spec['online_windows'],updates=updates,online_compute_wall_sec=wall,
            sum_squared_error=sums[0],sum_absolute_error=sums[1],complete=complete))
    checkpoint()
    for i in range(len(rows),spec['online_windows']):
        stop(checkpoint)
        row,elapsed=timed_observation(online_step,model,gate,data['stream'],i,updates)
        wall+=elapsed
        updates=row['cumulative_update_count'];rows.append(row)
        sums[0]+=row['squared_error'];sums[1]+=row['absolute_error']
        if len(rows)%250==0:checkpoint()
    return checkpoint(True)


def result(node,ck,protocol_hash,warmup_hash,calibration_hash=None):
    rows=ck['rows'];method={}
    if node['variant']=='fc':
        state=ck['state']['method']
        for key in ('raw_consistency_loss','weighted_consistency_loss'):
            values=[r['diagnostics'][key] for r in rows if r['update_flag']]
            if values:method['mean_'+key]=float(np.mean(values))
        method.update(EMA_update_count=state['teacher_update_count'],
            final_teacher_student_parameter_l2=runtime.norm([state['student'][k].double()-state['teacher'][k].double() for k in state['student']]))
    value=dict(dataset=node['dataset'],seed=node['seed'],variant=node['variant'],
        mode='r50' if node['variant']=='fc' else 'always',MSE=float(np.mean([r['squared_error'] for r in rows])),
        MAE=float(np.mean([r['absolute_error'] for r in rows])),updates=ck['updates'],eligible_online_steps=ck['n'],
        update_fraction=ck['updates']/ck['n'],online_compute_wall_sec=ck['online_compute_wall_sec'],
        protocol_hash=protocol_hash,warmup_sha256=warmup_hash,calibration_sha256=calibration_hash,
        completion_status='COMPLETE',method_diagnostics=method)
    if 'rho' in node:value['rho_target']=node['rho'];value['lambda_consistency']=ck['state']['method']['lambda_consistency']
    return value
