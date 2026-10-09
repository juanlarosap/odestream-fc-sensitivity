"""Small synthetic protocol tests. Never runs a dataset scientific trajectory."""
import copy
import contextlib
import importlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from experiments.odestream_fc_multidataset_v3 import engine as v3_engine, runtime, fc
from experiments.odestream_fc_multidataset_v3 import config as v3
from experiments.odestream_fc_multidataset_v3.model_adapter import Model, online_boundary
from experiments.odestream_fc_multidataset_v3.r50 import Gate
from experiments.odestream_fc_multidataset_v3.datasets import batch
from experiments.odestream_fc_multidataset_v3.losses import equal
from experiments.odestream_fc_multidataset_v3.checkpointing import save_checkpoint, load_checkpoint, sha, write_json, read_json
from experiments.odestream_fc_multidataset_v3.calibration import derive
from . import config as c, engine, provenance
from .run import Runner, parser, trace_bytes


def node(name, kind='online'):
    return next(n for n in c.master_plan()['nodes'] if n['dataset']=='ECL' and n['seed']==0
                and n['condition']==name and n['kind']==kind)


def tiny_data(count=4):
    # Actual v3 model, reduced synthetic time grid only in tests; no scientific loader changes.
    x = torch.linspace(-.2, .4, count*2).reshape(count, 2, 1)
    y = torch.linspace(.1, .3, count).reshape(count, 1)
    t = torch.tensor([0., .01]).reshape(1, 2, 1).repeat(count, 1, 1)
    return x, y, t


def model(name):
    runtime.seed(3)
    enabled = c.condition(name)['fc_enabled']
    return Model(1, 'fc' if enabled else 'control', 'online', .7 if enabled else None,
                 dict(calibration_identity='test', rho_target=1.) if enabled else {})


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runtime.configure()
        folder = v3.ROOT/'experiments/odestream_fc_multidataset_v3'
        cls.v3_hashes = {p:sha(p) for p in folder.iterdir() if p.is_file()}

    @classmethod
    def tearDownClass(cls):
        assert cls.v3_hashes == {p:sha(p) for p in cls.v3_hashes}, 'v3 files changed'

    def test_plan_dispatch_and_filters(self):
        plan = c.master_plan()
        self.assertEqual(plan['counts'], dict(online=30, warmups=30, bootstraps=15, calibration=0))
        self.assertEqual(len(plan['nodes']), 75)
        self.assertEqual(len({n['id'] for n in plan['nodes']}), 75)
        self.assertEqual({n['condition'] for n in plan['nodes']}, {'r50_only','fc_only'})
        self.assertEqual({n.get('rho_target') for n in plan['nodes']}, {None,1.0})
        for n in plan['nodes']:
            self.assertNotIn('variant', n)
            self.assertNotIn('rho', n)
            self.assertEqual(n['fc_enabled'], n['condition']=='fc_only')
            self.assertEqual(n['gate_mode'], 'always' if n['fc_enabled'] else 'r50')
            self.assertNotIn('calibration', n['parents'])
        selected = c.selection(plan, 'ETTh2', 0, ['r50_only','fc_only'])
        self.assertEqual(sum(n['kind']=='online' for n in selected), 2)
        for args in (['run','--rho','0.5'], ['run','--conditions','control'], ['run','--seed','5']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser().parse_args(args)
        malformed = dict(node('fc_only'), rho_target=.5)
        with self.assertRaises(ValueError):
            c.internal_node(malformed)

    def test_aliases_and_import_closure(self):
        self.assertIs(engine.warmup, v3_engine.warmup)
        self.assertIs(engine.bootstrap, v3_engine.bootstrap)
        self.assertIs(engine.online_step, v3_engine.online_step)
        from experiments.odestream_fc_multidataset_v3.timing import timed_observation
        self.assertIs(engine.timed_observation, timed_observation)
        manifest = provenance.source_manifest()
        self.assertTrue(all('corruption' not in path for path in manifest))
        self.assertFalse(any(path.endswith('/runner.py') for path in manifest))
        self.assertTrue(any(path.endswith('/test_protocol.py') for path in manifest))

    def test_r50_has_no_fc_or_coefficient_dependency(self):
        with patch('experiments.odestream_fc_multidataset_v3.model_adapter.FunctionalConsistencyState',
                   side_effect=AssertionError('teacher forbidden')):
            adapter = model('r50_only')
        self.assertIsNone(adapter.method)
        self.assertIsNone(adapter.coefficient)
        runner = Runner(Path('/unused-ablation'), Path('/nonexistent-v3'), {})
        self.assertEqual(runner.coefficient(node('r50_only')), (None, {}))
        self.assertTrue(any(n['kind']=='bootstrap' for n in c.selection(c.master_plan(), conditions=['r50_only'])))
        self.assertFalse(any(n['kind']=='bootstrap' for n in c.selection(c.master_plan(), conditions=['fc_only'])))

    def test_rejected_r50_exact_state_and_rng(self):
        adapter = model('r50_only')
        # Seed nonempty Adam state first, so zero_grad semantics are exercised.
        current = batch(tiny_data(), 0)
        adapter.update(current)
        before = adapter.snapshot()
        gate = Gate('r50', [1e12]*256)
        with patch.object(adapter, 'update', side_effect=AssertionError('rejected update')):
            row = engine.online_step(adapter, gate, current, 0, 0)
        after = adapter.snapshot()
        self.assertEqual(row['update_flag'], 0)
        self.assertEqual(row['diagnostics'], {})
        self.assertTrue(equal(before['model'], after['model']))
        self.assertTrue(equal(before['optimizer'], after['optimizer']))
        self.assertIsNone(after['method'])
        self.assertEqual(gate.state()[-1], row['squared_error'])
        self.assertFalse(equal(before['rng']['torch'], after['rng']['torch']))
        adapter.restore(before)
        adapter.forward(current)
        self.assertTrue(equal(runtime.rng(), after['rng']))

    def test_base_accepted_parity(self):
        adapter = model('r50_only')
        initial = adapter.snapshot()
        current = batch(tiny_data(), 0)
        row = engine.online_step(adapter, Gate('r50', [0.]*256), current, 0, 0)
        actual = adapter.snapshot()
        adapter.restore(initial)
        diagnostics = adapter.update(current)
        expected = adapter.snapshot()
        self.assertEqual(row['update_flag'], 1)
        self.assertEqual(row['diagnostics'], diagnostics)
        self.assertTrue(equal(actual, expected))
        self.assertEqual(set(diagnostics), {'mse_loss','kl_loss','l1_loss','total_loss'})

    def test_fc_always_parity_epsilon_and_ema_order(self):
        adapter = model('fc_only')
        initial = adapter.snapshot()
        self.assertTrue(equal(initial['model'], initial['method']['teacher']))
        current = batch(tiny_data(), 0)
        forwards, events = [], []
        original_forward, original_ema, original_step = fc.stochastic_forward, fc.update_ema_teacher, adapter.opt.step
        def forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            forwards.append(output[1])
            return output
        def step(*args, **kwargs):
            events.append('Adam')
            return original_step(*args, **kwargs)
        def ema(*args, **kwargs):
            events.append('EMA')
            return original_ema(*args, **kwargs)
        with patch.object(fc, 'stochastic_forward', side_effect=forward), \
             patch.object(adapter.opt, 'step', side_effect=step), \
             patch.object(fc, 'update_ema_teacher', side_effect=ema):
            # Adapter's imported forward must be observed too, without changing production bindings.
            with patch('experiments.odestream_fc_multidataset_v3.model_adapter.stochastic_forward', side_effect=forward):
                row = engine.online_step(adapter, Gate('always'), current, 0, 0)
        actual = adapter.snapshot()
        self.assertIs(forwards[0], forwards[1])
        self.assertEqual(events, ['Adam','EMA'])
        self.assertEqual(row['update_flag'], 1)
        self.assertEqual(adapter.method.teacher_update_count, 1)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in adapter.method.teacher.parameters()))
        for key in actual['model']:
            expected = .99*initial['method']['teacher'][key]+(1-.99)*actual['model'][key]
            self.assertTrue(torch.equal(actual['method']['teacher'][key], expected))
        adapter.restore(initial)
        diagnostics, _ = fc.functional_consistency_training_step(adapter.method, adapter.opt, current)
        self.assertEqual(row['diagnostics'], diagnostics)
        self.assertTrue(equal(actual, adapter.snapshot()))

    def test_online_phase_reset(self):
        for name in c.CONDITIONS:
            runtime.seed(0)
            enabled = c.condition(name)['fc_enabled']
            adapter = Model(1, 'fc' if enabled else 'control', 'warmup', .7 if enabled else None)
            adapter.update(batch(tiny_data(), 0))
            vae = copy.deepcopy(adapter.model.state_dict())
            transitioned = online_boundary(adapter)
            self.assertFalse(transitioned['optimizer']['state'])
            embedded = {k.removeprefix('model1.'):v for k,v in transitioned['model'].items() if k.startswith('model1.')}
            self.assertTrue(equal(vae, embedded))
            if enabled:
                self.assertTrue(equal(transitioned['model'], transitioned['method']['teacher']))
                self.assertEqual(transitioned['method']['teacher_update_count'], 0)
            else:
                self.assertIsNone(transitioned['method'])

    def test_bootstrap_restores_rng(self):
        adapter = model('r50_only')
        initial = dict(state=adapter.snapshot())
        runtime.seed(17)  # bootstrap must restore initial checkpoint RNG, not caller's unrelated RNG
        record = engine.bootstrap(c.internal_node(node('r50_only','bootstrap')), dict(input_dim=1, validation=[0,256]),
                                  dict(validation=tiny_data(256)), initial, copy.deepcopy, lambda:None)
        self.assertEqual(len(record['errors']), 256)
        self.assertTrue(equal(runtime.rng(), initial['state']['rng']))
        self.assertTrue(equal(adapter.snapshot()['model'], initial['state']['model']))
        second = engine.bootstrap(c.internal_node(node('r50_only','bootstrap')), dict(input_dim=1, validation=[0,256]),
                                  dict(validation=tiny_data(256)), initial, copy.deepcopy, lambda:None)
        self.assertTrue(equal(record, second))

    def test_resume_serialized_state_and_timer(self):
        for name in c.CONDITIONS:
            with self.subTest(condition=name), tempfile.TemporaryDirectory() as directory:
                adapter = model(name)
                initial = dict(state=adapter.snapshot())
                coefficient, prov = adapter.coefficient, adapter.provenance
                spec, data = dict(input_dim=1, online_windows=4), dict(stream=tiny_data())
                boot = [.001]*256 if name=='r50_only' else []
                args = (node(name), spec, data, initial, boot, coefficient, prov)
                original_timer = engine.timed_observation
                with patch.object(engine, 'timed_observation', wraps=original_timer) as timer:
                    full = engine.online(*args, None, copy.deepcopy, lambda checkpoint=None:None)
                self.assertEqual(timer.call_count, 4)
                self.assertTrue(all(call.args[0] is v3_engine.online_step for call in timer.call_args_list))
                path = Path(directory)/'state.pt'
                def commit(value):
                    save_checkpoint(path, value)
                    return copy.deepcopy(value)
                calls = 0
                def stop(checkpoint=None):
                    nonlocal calls
                    calls += 1
                    if calls == 3:
                        checkpoint()
                        raise InterruptedError('synthetic interruption')
                with self.assertRaises(InterruptedError):
                    engine.online(*args, None, commit, stop)
                saved = load_checkpoint(path)
                self.assertEqual(saved['next_index'], 2)
                resumed = engine.online(*args, saved, commit, lambda checkpoint=None:None)
                for key in full:
                    if key != 'online_compute_wall_sec':
                        self.assertTrue(equal(full[key], resumed[key]), key)
                engine.validate_online(resumed, node(name), spec, boot, data)
                if name=='fc_only':
                    self.assertIsNone(resumed['window'])
                    self.assertEqual(resumed['updates'], 4)
                    self.assertEqual(resumed['state']['method']['teacher_update_count'], 4)
                    self.assertEqual(engine.result(node(name), resumed, 'test', 'warm')['update_fraction'], 1.)
                with self.assertRaises(ValueError):
                    corrupted = copy.deepcopy(resumed)
                    corrupted['rows'][0]['update_flag'] = 1-corrupted['rows'][0]['update_flag']
                    engine.validate_online(corrupted, node(name), spec, boot, data)

    def test_output_protection(self):
        for path in (v3.ROOT, v3.ROOT/'experiments/new', v3.ROOT/'data/new', c.DEFAULT_V3_OUTPUT,
                     c.DEFAULT_V3_OUTPUT/'nested', c.DEFAULT_V3_OUTPUT.parent):
            with self.assertRaises(ValueError):
                c.safe_output(path)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'ablation'
            self.assertEqual(c.safe_output(out), out.resolve())
            out.mkdir()
            write_json(out/'protocol.json', dict(experiment_id='wrong'))
            with self.assertRaises(ValueError):
                c.safe_output(out)

    def test_r50_preflight_never_reads_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'ablation'
            runner = Runner(out, Path(directory)/'absent-v3',
                            dict(dataset='ECL', seed=0, conditions=['r50_only']), True)
            with patch('experiments.odestream_fc_ablation_v1.run.verify_references', return_value={}) as refs, \
                 patch('experiments.odestream_fc_ablation_v1.run.verify_datasets'), \
                 patch('experiments.odestream_fc_multidataset_v3.runtime.configure'), \
                 patch('experiments.odestream_fc_multidataset_v3.datasets.load_data', return_value=({},{})):
                runner.preflight()
            refs.assert_called_once_with((Path(directory)/'absent-v3').resolve(), [])
            self.assertFalse((out/'datasets/ECL/calibration_reference.json').exists())

    def test_warmup_resume_both_families(self):
        # Synthetic 64-window dataset, short integration span and forced equal
        # validation scores: exercises the real 10-epoch patience path cheaply.
        def windows(count):
            x = torch.linspace(-.2,.4,count*24).reshape(count,24,1)
            y = torch.ones(count,1)*.2
            t = torch.linspace(0.,.01,24).reshape(1,24,1).repeat(count,1,1)
            return x,y,t
        data = dict(train=windows(64), validation=windows(2))
        spec = dict(input_dim=1, train_windows=64)
        original_validation = v3_engine.validation
        def fixed_score(*args):
            original_validation(*args)  # preserve actual validation RNG consumption
            return 1.
        for name in c.CONDITIONS:
            with self.subTest(condition=name), tempfile.TemporaryDirectory() as directory:
                n = c.internal_node(node(name,'warmup'))
                coefficient = .7 if n['fc_enabled'] else None
                args = (n, spec, data, coefficient, {})
                path = Path(directory)/'warmup.pt'
                def commit(value):
                    save_checkpoint(path,value)
                    return copy.deepcopy(value)
                calls = 0
                def stop(checkpoint=None):
                    nonlocal calls
                    calls += 1
                    if calls == 3:
                        checkpoint()
                        raise InterruptedError('warmup interruption')
                with patch.object(v3_engine,'validation',side_effect=fixed_score):
                    full = engine.warmup(*args,None,copy.deepcopy,lambda checkpoint=None:None)
                    with self.assertRaises(InterruptedError):
                        engine.warmup(*args,None,commit,stop)
                    saved = load_checkpoint(path)
                    v3_engine.validate_warmup(saved,n,spec)
                    resumed = engine.warmup(*args,saved,commit,lambda checkpoint=None:None)
                v3_engine.validate_warmup(resumed,n,spec)
                self.assertEqual(resumed['epoch'],11)
                for key in full:
                    if key!='warmup_seconds':
                        self.assertTrue(equal(full[key],resumed[key]),key)

    def test_summary_sample_sd_and_gate_ties(self):
        from experiments.odestream_fc_multidataset_v3.reporting import summary
        value = summary([1.,2.,3.,4.,5.])
        self.assertEqual(value['mean'],3.)
        self.assertAlmostEqual(value['sample_sd'],2.5**.5)
        self.assertEqual(value['seeds'],[1.,2.,3.,4.,5.])
        gate = Gate('r50',[2.]*256)
        threshold = gate.threshold()
        self.assertEqual(gate.decide(2.,threshold),1)
        self.assertEqual(gate.decide(1.,threshold),0)
        gate.append(1.)
        self.assertEqual(gate.state()[-1],1.)

    def test_runner_online_exports_recovery_and_tamper(self):
        for name in c.CONDITIONS:
            with self.subTest(condition=name), tempfile.TemporaryDirectory() as directory:
                out = Path(directory)/'ablation'
                runner = Runner(out, Path(directory)/'no-v3',
                                dict(dataset='ECL',seed=0,conditions=[name]),True)
                runner.specs['ECL'] = dict(runner.specs['ECL'], input_dim=1,online_windows=4)
                runner.data['ECL'] = dict(stream=tiny_data())
                n = node(name)
                adapter = model(name)
                initial = dict(state=adapter.snapshot())
                if n['fc_enabled']:
                    runner.references['ECL'] = dict(lambda_consistency=.7,coefficient_sha256='test',artifacts={})
                    write_json(out/'datasets/ECL/calibration_reference.json',runner.references['ECL'])
                for filename,value in [('protocol.json',c.protocol()),('plan.json',runner.plan),('sources.json',runner.sources)]:
                    write_json(out/filename,value)
                boot = [0.]*256 if name=='r50_only' else []
                # Parent fixture isolates orchestration/persistence. Actual warmup
                # and bootstrap mathematics are exercised in separate tests above.
                for identifier in n['parents']:
                    parent = runner.lookup[identifier]
                    payload = dict(initial,complete=True) if parent['kind']=='warmup' else dict(errors=boot,complete=True)
                    save_checkpoint(runner.path(parent),payload)
                    rel = str(runner.path(parent).relative_to(out))
                    write_json(runner.receipt(parent),dict(protocol_hash=runner.hash,node_id=identifier,
                               parents={},artifacts={rel:sha(runner.path(parent))}))
                coefficient,prov = runner.coefficient(n)
                ck = engine.online(n,runner.specs['ECL'],runner.data['ECL'],initial,boot,coefficient,prov,None,
                                   lambda value:runner.commit(n,value),lambda checkpoint=None:None)
                runner.validate(n,ck)
                self.assertEqual(runner.state['jobs'][n['id']]['status'],'COMPLETE')
                value = read_json(out/n['id']/'result.json')
                self.assertEqual(value['condition'],name)
                self.assertTrue(value['non_authoritative'])
                self.assertNotIn('variant',value)
                self.assertEqual((out/n['id']/'trace.csv.gz').read_bytes(),trace_bytes(ck['rows']))
                with patch.object(engine,'online',side_effect=AssertionError('completed trajectory replayed')):
                    runner.run_node(n)
                # Crash after state.pt but before derived export/receipt: recover,
                # without an additional online step.
                runner.receipt(n).unlink()
                (out/n['id']/'result.json').unlink()
                (out/n['id']/'trace.csv.gz').unlink()
                with patch.object(engine,'online',side_effect=AssertionError('recovery replayed science')):
                    runner.run_node(n)
                value['MSE'] += 1
                write_json(out/n['id']/'result.json',value)
                with self.assertRaises(ValueError):
                    runner.finish(n,ck)

    def test_standalone_report_shape(self):
        from . import reporting
        rows = []
        for d in c.DATASETS:
            for name in c.CONDITIONS:
                for seed in c.SEEDS:
                    rows.append(dict(dataset=d,condition=name,seed=seed,non_authoritative=True,
                                     MSE=float(seed+1),MAE=float(seed+1),online_compute_wall_sec=float(seed+1),
                                     update_fraction=1.,updates=4,eligible_online_steps=4))
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(reporting,'collect',return_value=rows):
                artifacts = reporting.aggregate(Path(directory))
            self.assertEqual(len(artifacts),3)
            report = read_json(Path(directory)/'reports/aggregate.json')
            self.assertTrue(report['non_authoritative'])
            self.assertEqual(report['seed_order'],list(c.SEEDS))
            self.assertEqual(set(report['datasets']),set(c.DATASETS))
            self.assertEqual(set(report['datasets']['ECL']),set(c.CONDITIONS))
            self.assertEqual(report['datasets']['ECL']['fc_only']['MSE']['mean'],3.)
            self.assertAlmostEqual(report['datasets']['ECL']['fc_only']['MSE']['sample_sd'],2.5**.5)

    def test_calibration_verification_and_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            self.calibration_fixture(out)
            before = {str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
            record = provenance.verify_calibration(out, 'ECL')
            self.assertEqual(record['rho_target'], 1.)
            self.assertEqual(record['lambda_consistency'], 2.)
            self.assertEqual(before, {str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
            # Extra historical files in original source inventory are harmless.
            self.assertIn('historical/unrelated.py', read_json(out/'sources.json'))
            coefficient = out/'datasets/ECL/calibration/rho_1.0.json'
            value = read_json(coefficient)
            value['rho_target'] = .5
            write_json(coefficient, value)
            with self.assertRaises(ValueError):
                provenance.verify_calibration(out, 'ECL')
            self.calibration_fixture(out)
            source = read_json(out/'sources.json')
            source['experiments/odestream_fc_multidataset_v3/fc.py'] = 'wrong'
            write_json(out/'sources.json', source)
            with self.assertRaises(ValueError):
                provenance.verify_calibration(out, 'ECL')
            with self.assertRaises(ValueError):
                provenance.verify_references(out, ['ETTh2','Weather'])

    @staticmethod
    def calibration_fixture(out):
        sources = {f'experiments/odestream_fc_multidataset_v3/{name}':sha(v3.ROOT/'experiments/odestream_fc_multidataset_v3'/name)
                   for name in provenance.V3_FILES}
        sources['historical/unrelated.py'] = 'a'*64
        write_json(out/'protocol.json', v3.protocol())
        write_json(out/'plan.json', v3.master_plan())
        write_json(out/'sources.json', sources)
        write_json(out/'execution_environment.json', {'fixture':True})
        ph = v3.digest(v3.protocol())
        lookup = {n['id']:n for n in v3.master_plan()['nodes']}
        prep = 'datasets/ECL/preparation/control_seed_0'
        (out/prep).mkdir(parents=True, exist_ok=True)
        (out/prep/'state.pt').write_bytes(b'fixture preparation; checked by checksum only')
        write_json(out/(prep+'.lineage.json'), dict(protocol_hash=ph, node_id=prep,
                   parents={}, artifacts={prep+'/state.pt':sha(out/prep/'state.pt')}))
        rows = [dict(G_base=1.-1e-12, G_cons=.5, unscaled_gradient_ratio=.5,
                     lambda_consistency=0., student_update_count=i+1, teacher_update_count=i+1) for i in range(128)]
        reference = dict(seed=0, steps=128, batch_size=1, beta=.99, eps=1e-12, lambda_zero=True,
                         exclusion_tolerance=0., target_rows=[24,152], input_row_union=[0,151],
                         all_128_baseline_commits_exact=True, valid_ratio_count=128,
                         exact_zero_consistency_gradient_steps=0, median_unscaled_gradient_ratio=.5, rows=rows,
                         initialization='separate validation-selected preparation Control warmup',
                         gradient_observations='TRAIN only; no online observations', complete=True)
        for identifier, value, parent in (
                ('datasets/ECL/calibration/reference', reference, prep+'/state.pt'),
                ('datasets/ECL/calibration/rho_1.0', derive(reference, 1.), 'datasets/ECL/calibration/reference.json')):
            value = dict(value, protocol_hash=ph, source_hash=v3.digest(sources), node=lookup[identifier],
                         dataset_sha256=v3.registry()['ECL']['sha256'], parents={parent:sha(out/parent)})
            write_json(out/(identifier+'.json'), value)
            write_json(out/(identifier+'.lineage.json'), dict(protocol_hash=ph, node_id=identifier,
                       parents=value['parents'], artifacts={identifier+'.json':sha(out/(identifier+'.json'))}))


if __name__ == '__main__':
    unittest.main()
