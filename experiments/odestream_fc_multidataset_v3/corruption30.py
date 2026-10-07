"""Raw-zero input corruption with clean labels; no scientific execution at import.

One mask per dataset/seed, before scaling and overlapping windows. This is not
irregular sampling. All feature histories, including the target feature, are
eligible. Only the label role uses the clean view.
"""
import copy
import hashlib
import json
from pathlib import Path

from . import config as v3
from .checkpointing import atomic, read_json, sha, write_json

EXPERIMENT_ID = 'odestream-fc-input-corruption30-v1'
DEFAULT_OUTPUT = v3.ROOT / 'outputs' / EXPERIMENT_ID
NEW_FILES = ('corruption30.py', 'run_corruption30.py', 'report_corruption30.py')


def protocol():
    value = copy.deepcopy(v3.protocol())
    value.update(version=1, experiment_id=EXPERIMENT_ID, base_protocol='odestream-fc-sensitivity-v3',
                 rho_targets=[1.0], condition_order=['control/always', 'fc/rho_1.0/r50'],
                 scientific_warmups=30, scientific_online_trajectories=30,
                 scaler='Corrupted TRAIN inputs only; clean labels use the same target-feature mean/scale')
    value['corruption'] = dict(fraction=0.30, count='floor(3*N*F/10)', replacement_raw=0.0,
        eligibility='all raw row x model-input-feature cells, including historical target feature',
        labels='clean next recorded target, normalized by corrupted TRAIN input scaler',
        scope=['train', 'validation', 'bootstrap', 'calibration', 'stream'],
        rng='isolated numpy.random.RandomState (MT19937); choice without replacement',
        rng_seed='300000 + 100*dataset_index + scientific_seed; dataset_index=ETTh2:0,ECL:1,Weather:2',
        pairing='one persisted authoritative mask per dataset/seed, shared across methods',
        windows='persistent raw-cell corruption; never redraw per window',
        timestamps='retain exact rows and timestamps; v3 artificial integration grid unchanged')
    value['calibration'].update(data='seed-0 mask; corrupted inputs and clean labels',
        lambda_scope='one recalibrated dataset coefficient shared across scientific seeds 0-4')
    for spec in value['datasets'].values():
        spec['row_policy'] = 'preserve recorded rows/order; separate clean labels and zero-corrupted input view'
    return value


def master_plan():
    nodes = copy.deepcopy(v3.selection(v3.master_plan(), rho=1.0))
    online = [node for node in nodes if node['kind'] == 'online']
    assert len(online) == 30 and sum(n['kind'] == 'warmup' and n['scientific'] for n in nodes) == 30
    return dict(protocol_hash=v3.digest(protocol()), nodes=nodes, online_jobs=online,
                counts=dict(datasets=3, seeds=5, rhos=1, control_online=15, fc_online=15,
                            online=30, scientific_warmups=30))


def safe_output(path):
    """Only dedicated repository-relative outputs; never a clean-output alias."""
    path = Path(path)
    out = (v3.ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    outputs, clean = (v3.ROOT / 'outputs').resolve(), v3.DEFAULT_OUTPUT.resolve()
    if (not out.is_relative_to(outputs) or out == outputs or out == clean
            or out.is_relative_to(clean) or clean.is_relative_to(out)):
        raise ValueError('Use a dedicated corruption output directory under outputs/, outside clean v3 outputs')
    if out.exists():
        if any(p.is_symlink() for p in out.rglob('*')):
            raise ValueError('Output artifacts must not be symlinks')
        for name in ('protocol.json', 'state.json'):
            if (out / name).exists() and read_json(out / name).get('experiment_id') != EXPERIMENT_ID:
                raise ValueError('Output belongs to another experiment')
        if any(out.iterdir()) and not (out / 'protocol.json').exists():
            raise ValueError('Nonempty output without corruption protocol; investigate before running')
    return out


def pair_key(dataset, seed):
    if dataset not in v3.DATASETS or seed not in v3.SEEDS:
        raise ValueError('Unknown dataset/seed')
    return f'{dataset}/seed_{seed}'


def mask_paths(out, dataset, seed):
    base = Path(out) / 'masks' / pair_key(dataset, seed)
    return base / 'mask.npz', base / 'manifest.json'


def read_clean(dataset):
    """Validate v3 raw schema/time contracts without constructing clean windows."""
    import numpy as np
    import pandas as pd
    spec = v3.registry()[dataset]
    path = v3.ROOT / spec['file']
    if sha(path) != spec['sha256']:
        raise ValueError('Dataset hash mismatch: ' + dataset)
    frame = pd.read_csv(path, encoding='utf-8')
    assert frame.columns.tolist() == ['date'] + spec['numerical_columns']
    assert frame.shape == (spec['rows'], len(spec['numerical_columns']) + 1)
    dates = pd.to_datetime(frame.date, format='%Y-%m-%d %H:%M:%S', errors='raise')
    assert dates.notna().all() and dates.is_monotonic_increasing
    delta = dates.diff().dt.total_seconds()
    if dataset == 'Weather':
        assert path.read_bytes().decode('utf-8').encode('utf-8') == path.read_bytes()
        assert list(frame.index[delta.notna() & delta.ne(600)]) == [a['row'] for a in v3.ANOMALIES]
        for a in v3.ANOMALIES:
            assert str(dates.iloc[a['previous_row']]) == a['previous']
            assert str(dates.iloc[a['row']]) == a['current'] and delta.iloc[a['row']] == a['seconds']
        assert list(frame.index[dates.duplicated()]) == [19044]
        assert list(frame.index[frame.duplicated()]) == [19044]
        assert frame.iloc[19043].equals(frame.iloc[19044])
        assert not frame.isna().any().any() and not any(frame[c].nunique(dropna=False) == 1 for c in frame)
    else:
        assert dates.is_unique and delta.dropna().eq(3600).all()
    assert all(pd.api.types.is_numeric_dtype(t) for t in frame.iloc[:, 1:].dtypes)
    clean = frame[spec['inputs']].to_numpy(dtype=float)
    assert np.isfinite(clean).all()
    assert spec['online'][0] == len(clean) // 4 and spec['train'][1] == spec['online'][0] * 4 // 5
    return clean


def array_hash(array):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_mask(out, dataset, seed, clean, create=False):
    import numpy as np
    spec = v3.registry()[dataset]
    path, manifest_path = mask_paths(out, dataset, seed)
    eligible = clean.size
    rng_seed = 300000 + 100 * v3.DATASETS.index(dataset) + seed
    expected = dict(dataset=dataset, scientific_seed=seed, corruption_rng_seed=rng_seed,
        raw_dataset_sha256=spec['sha256'], clean_array_shape=list(clean.shape),
        model_input_features=spec['inputs'], eligible_cells=eligible,
        selected_cells=3 * eligible // 10, requested_fraction=0.30,
        rng='numpy.random.RandomState/MT19937', coordinates='sorted C-order flat indices; row=i//F, feature=i%F')
    if path.exists():
        with np.load(path, allow_pickle=False) as artifact:
            indices = artifact['indices']
            record = json.loads(str(artifact['metadata'].item()))
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError('Mask identity mismatch: ' + pair_key(dataset, seed))
    else:
        if not create or manifest_path.exists() or (Path(out) / 'data_identities.json').exists():
            raise ValueError('Authoritative mask missing: ' + pair_key(dataset, seed))
        indices = np.sort(np.random.RandomState(rng_seed).choice(eligible, expected['selected_cells'], replace=False)).astype('<i8')
        record = dict(expected, numpy_version=np.__version__)
    if (indices.dtype != np.dtype('<i8') or indices.shape != (expected['selected_cells'],)
            or (len(indices) and (indices[0] < 0 or indices[-1] >= eligible or not np.all(np.diff(indices) > 0)))):
        raise ValueError('Invalid mask coordinates')
    already_zero = int(np.count_nonzero(clean.reshape(-1)[indices] == 0))
    stats = dict(mask_sha256=array_hash(indices), realized_mask_fraction=len(indices) / eligible,
        selected_original_zeros=already_zero, changed_cells=len(indices) - already_zero,
        effective_changed_value_fraction=(len(indices) - already_zero) / eligible)
    if path.exists():
        if any(record.get(k) != v for k, v in stats.items()):
            raise ValueError('Mask checksum/statistics mismatch')
    else:
        record.update(stats)
        atomic(path, lambda f: np.savez_compressed(f, indices=indices,
               metadata=np.array(json.dumps(record, sort_keys=True, allow_nan=False))), True)
    return indices, record, sha(path)


def prepare_pair(out, dataset, seed, clean=None, create=False, windows=True):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    spec = v3.registry()[dataset]
    clean = read_clean(dataset) if clean is None else clean
    assert clean.shape == (spec['rows'], spec['input_dim']) and np.isfinite(clean).all()
    indices, mask, artifact_hash = load_mask(out, dataset, seed, clean, create)
    corrupted = clean.copy(order='C')
    corrupted.reshape(-1)[indices] = 0.0
    train, offline = spec['train'][1], spec['online'][0]
    scaler = StandardScaler().fit(corrupted[:train])
    scaled = scaler.transform(corrupted)
    target = spec['inputs'].index(spec['target'])
    labels = (clean[:, target:target+1] - scaler.mean_[target]) / scaler.scale_[target]
    np.testing.assert_array_equal(scaler.mean_, corrupted[:train].mean(axis=0))
    assert int(scaler.n_samples_seen_) == train
    assert np.isfinite(scaled).all() and np.isfinite(labels).all()
    scaler_record = dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                         var=scaler.var_.tolist(), n_samples_seen=int(scaler.n_samples_seen_))
    manifest = dict(dataset=dataset, scientific_seed=seed, raw_dataset_sha256=spec['sha256'],
        mask=mask, mask_artifact_sha256=artifact_hash, scaler=scaler_record,
        scaler_sha256=v3.digest(scaler_record), input_scaled_sha256=array_hash(scaled),
        clean_scaled_target_sha256=array_hash(labels), target=spec['target'],
        train=spec['train'], validation=spec['validation'], online=spec['online'])
    manifest['data_identity'] = v3.digest(manifest)
    _, manifest_path = mask_paths(out, dataset, seed)
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            raise ValueError('Frozen scaler/data identity mismatch: ' + pair_key(dataset, seed))
    elif create:
        write_json(manifest_path, manifest)
    else:
        raise ValueError('Pair manifest missing')
    if not windows:
        return None, manifest
    import torch
    def make(start, end):
        n = end - start - 24
        x = torch.tensor(np.array([scaled[i:i+24] for i in range(start, end-24)]), dtype=torch.float32)
        y = torch.tensor(labels[start+24:end], dtype=torch.float32)
        grid = np.hstack([np.linspace(0, 24, num=24)[:, None]] * n)
        t = torch.from_numpy(grid[:, :, None]).to(torch.float32).transpose(0, 1)
        # Check every overlapping window, not just a representative first batch.
        for offset in range(24):
            np.testing.assert_array_equal(x[:, offset].numpy(), scaled[start+offset:start+offset+n].astype(np.float32))
        np.testing.assert_array_equal(y.numpy(), labels[start+24:end].astype(np.float32))
        assert x.shape == (n, 24, spec['input_dim']) and y.shape == (n, 1) and t.shape == (n, 24, 1)
        return x, y, t
    data = dict(train=make(0, train), validation=make(train-24, offline), stream=make(offline-24, len(clean)))
    for phase in data:
        expected = spec['online_windows'] if phase == 'stream' else spec[phase + '_windows']
        assert len(data[phase][0]) == expected
    return data, manifest
