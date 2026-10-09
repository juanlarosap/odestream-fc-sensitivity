"""Explicit gate wrapper; all timed scientific operations are the original v3 calls."""
import numpy as np
from experiments.odestream_fc_multidataset_v3 import engine as v3
from experiments.odestream_fc_multidataset_v3 import runtime
from experiments.odestream_fc_multidataset_v3.losses import equal, finite
from experiments.odestream_fc_multidataset_v3.model_adapter import Model
from experiments.odestream_fc_multidataset_v3.r50 import Gate
from experiments.odestream_fc_multidataset_v3.timing import timed_observation
from .config import internal_node

# Direct aliases: no altered warmup, bootstrap, forward, loss, backward or timing.
warmup = v3.warmup
bootstrap = v3.bootstrap
online_step = v3.online_step


def online(node, spec, data, initial, boot, coefficient, provenance, saved, commit, stop):
    internal = internal_node(node)
    if not node['fc_enabled'] and (coefficient is not None or provenance):
        raise ValueError('R50-only must have no lambda dependency')
    model = Model(spec['input_dim'], internal['variant'], 'online', coefficient, provenance,
                  saved['state'] if saved is not None else initial['state'])
    rows = [] if saved is None else saved['rows']
    wall = 0. if saved is None else saved['online_compute_wall_sec']
    updates = 0 if saved is None else saved['updates']
    gate = Gate(node['gate_mode'], boot if saved is None else (saved['window'] or []))
    sums = [sum(r['squared_error'] for r in rows), sum(r['absolute_error'] for r in rows)]

    def checkpoint(complete=False):
        return commit(dict(state=model.snapshot(), rows=rows, window=gate.state(), next_index=len(rows),
                           n=spec['online_windows'], updates=updates, online_compute_wall_sec=wall,
                           sum_squared_error=sums[0], sum_absolute_error=sums[1], complete=complete))

    checkpoint()
    for i in range(len(rows), spec['online_windows']):
        stop(checkpoint)
        row, elapsed = timed_observation(online_step, model, gate, data['stream'], i, updates)
        wall += elapsed
        updates = row['cumulative_update_count']
        rows.append(row)
        sums[0] += row['squared_error']
        sums[1] += row['absolute_error']
        if len(rows) % 250 == 0:
            checkpoint()
    return checkpoint(True)


def validate_online(ck, node, spec, boot, data=None):
    """v3 validation contract with independent gate selection."""
    internal = internal_node(node)
    total = spec['online_windows']
    if ck['n'] != total or ck['next_index'] != len(ck['rows']) or not 0 <= ck['next_index'] <= total:
        raise ValueError('Online position mismatch')
    gate, updates = Gate(node['gate_mode'], boot), 0
    for i, row in enumerate(ck['rows']):
        error = (row['target']-row['prediction'])**2
        tau = gate.threshold()
        flag = gate.decide(error, tau)
        gate.append(error)
        updates += flag
        expected = dict(online_index=i, squared_error=error,
                        absolute_error=abs(row['target']-row['prediction']), gate_threshold=tau,
                        update_flag=flag, cumulative_update_count=updates, rolling_error_statistic=error)
        if any(row[k] != value for k, value in expected.items()):
            raise ValueError('Online trace/gate mismatch')
        if data is not None and row['target'] != data['stream'][1][i].item():
            raise ValueError('Trace target mismatch')
        if not flag and row['diagnostics']:
            raise ValueError('Skipped update diagnostics')
    if ck['updates'] != updates or ck['window'] != gate.state():
        raise ValueError('FIFO/update mismatch')
    if ck['sum_squared_error'] != sum(r['squared_error'] for r in ck['rows']):
        raise ValueError('MSE accumulator mismatch')
    if ck['sum_absolute_error'] != sum(r['absolute_error'] for r in ck['rows']):
        raise ValueError('MAE accumulator mismatch')
    if ck['complete'] and ck['next_index'] != total:
        raise ValueError('Premature completion')
    v3.validate_state(ck['state'], internal['variant'], updates)
    finite(ck['rows'], 'trace')
    finite(ck['online_compute_wall_sec'], 'online time')
    if ck['online_compute_wall_sec'] < 0:
        raise ValueError('Negative time')
    if node['fc_enabled']:
        if not equal(ck['state']['method']['rows'], [r['diagnostics'] for r in ck['rows'] if r['update_flag']]):
            raise ValueError('FC diagnostic history mismatch')
        if updates != ck['next_index'] or ck['window'] is not None:
            raise ValueError('FC-only must always update without a FIFO')


def result(node, ck, protocol_hash, warmup_hash, calibration_hash=None):
    internal_node(node)
    rows, method = ck['rows'], {}
    if not ck['complete'] or len(rows) != ck['n'] or not rows:
        raise ValueError('Result requires a complete trajectory')
    if node['fc_enabled']:
        state = ck['state']['method']
        for key in ('raw_consistency_loss', 'weighted_consistency_loss'):
            values = [r['diagnostics'][key] for r in rows if r['update_flag']]
            if values:
                method['mean_'+key] = float(np.mean(values))
        method.update(EMA_update_count=state['teacher_update_count'],
                      final_teacher_student_parameter_l2=runtime.norm([
                          state['student'][k].double()-state['teacher'][k].double() for k in state['student']]))
    value = dict(dataset=node['dataset'], seed=node['seed'], condition=node['condition'],
                 fc_enabled=node['fc_enabled'], gate_mode=node['gate_mode'],
                 MSE=float(np.mean([r['squared_error'] for r in rows])),
                 MAE=float(np.mean([r['absolute_error'] for r in rows])), updates=ck['updates'],
                 eligible_online_steps=ck['n'], update_fraction=ck['updates']/ck['n'],
                 online_compute_wall_sec=ck['online_compute_wall_sec'], protocol_hash=protocol_hash,
                 warmup_sha256=warmup_hash, calibration_sha256=calibration_hash,
                 completion_status='COMPLETE', method_diagnostics=method)
    if node['fc_enabled']:
        value.update(rho_target=1.0, lambda_consistency=ck['state']['method']['lambda_consistency'])
    return value
