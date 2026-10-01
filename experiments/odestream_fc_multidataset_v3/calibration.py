"""Lambda-zero TRAIN reference. Q95 is diagnostic, never a rho selection gate."""
import copy
import math
import numpy as np
from . import runtime
from .fc import FunctionalConsistencyState, functional_consistency_training_step, EPS
from .model_adapter import Model, combined
from .losses import optimizer, paperloss, equal, finite
from .datasets import batch


def calibrate(dim,data,initial,saved,commit,stop):
    student=Model(dim,'control','online',checkpoint=initial['state'])
    assert not student.opt.state
    state=FunctionalConsistencyState(student.model,0,'calibration')
    if saved is not None:
        state.load_state_dict(saved['fc']);student.opt.load_state_dict(saved['optimizer']);runtime.restore_rng(saved['rng'])
    baseline=combined(dim);baseline.load_state_dict(state.student.state_dict())
    opt=optimizer(baseline);opt.load_state_dict(copy.deepcopy(student.opt.state_dict()))
    runtime.restore_rng(saved['rng'] if saved is not None else initial['state']['rng'])
    def checkpoint():
        return dict(fc=state.state_dict(),optimizer=student.opt.state_dict(),rng=runtime.rng(),
                    next_index=state.student_update_count,complete=False)
    if saved is None:commit(checkpoint())
    for i in range(state.student_update_count,128):
        stop()
        current=batch(data['train'],i);before=runtime.rng()
        opt.zero_grad();output=baseline(current[0],current[1],None)
        loss=paperloss(current[2],output[0],*output[-3:]);loss['total_loss'].backward();opt.step()
        after=runtime.rng();runtime.restore_rng(before)
        row,pred=functional_consistency_training_step(state,student.opt,current,gradient_metrics=True)
        assert equal(pred,output[0].detach()) and equal(baseline.state_dict(),state.student.state_dict())
        assert equal(opt.state_dict(),student.opt.state_dict()) and equal(after,runtime.rng())
        commit(checkpoint())
    rows=state.rows;valid=[r for r in rows if r['G_cons']>0]
    ratios=[r['unscaled_gradient_ratio'] for r in valid]
    median=float(np.median(ratios)) if ratios else 0.
    if len(valid)<2 or not math.isfinite(median) or median<=0:raise ValueError('Invalid calibration reference')
    record=dict(seed=0,steps=128,batch_size=1,beta=.99,eps=EPS,lambda_zero=True,
        median_unscaled_gradient_ratio=median,valid_ratio_count=len(valid),
        exact_zero_consistency_gradient_steps=128-len(valid),
        near_zero_nonzero_gradient_steps=sum(0<r['G_cons']<=1e-12 for r in rows),
        exclusion_tolerance=0.,target_rows=[24,152],input_row_union=[0,151],
        initialization='separate validation-selected preparation Control warmup',
        gradient_observations='TRAIN only; no online observations',all_128_baseline_commits_exact=True,
        unscaled_gradient_ratio=runtime.stats(ratios),
        all_steps_unscaled_gradient_ratio=runtime.stats([r['unscaled_gradient_ratio'] for r in rows]),
        gradient_cosine=runtime.stats([r['gradient_cosine'] for r in rows if r['gradient_cosine'] is not None]),
        rows=rows,complete=True)
    finite(record,'calibration reference')
    return record


def derive(reference,rho):
    coefficient=rho/reference['median_unscaled_gradient_ratio']
    if not math.isfinite(coefficient) or coefficient<=0:raise ValueError('Invalid lambda')
    valid=[coefficient*r['unscaled_gradient_ratio'] for r in reference['rows'] if r['G_cons']>0]
    return dict(rho_target=rho,lambda_consistency=coefficient,
        scaled_gradient_ratios_valid=runtime.stats(valid),
        scaled_gradient_ratios_all_steps=runtime.stats([coefficient*r['unscaled_gradient_ratio'] for r in reference['rows']]),
        q95_policy='diagnostic only; no rejection, selection, ranking or clipping',complete=True)
