"""Exact dataset contracts. Loading/window preparation is explicit, never at import."""
from .config import ROOT, registry, ANOMALIES
from .checkpointing import sha


def load_data(name):
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.preprocessing import StandardScaler
    spec=registry()[name];path=ROOT/spec['file']
    if sha(path)!=spec['sha256']:raise ValueError('Dataset hash mismatch: '+name)
    frame=pd.read_csv(path,encoding='utf-8')
    assert frame.columns.tolist()==['date']+spec['numerical_columns']
    assert frame.shape==(spec['rows'],len(spec['numerical_columns'])+1)
    dates=pd.to_datetime(frame.date,format='%Y-%m-%d %H:%M:%S',errors='raise')
    assert dates.notna().all() and dates.is_monotonic_increasing
    delta=dates.diff().dt.total_seconds()
    if name=='Weather':
        assert path.read_bytes().decode('utf-8').encode('utf-8')==path.read_bytes()
        bad=list(frame.index[delta.notna() & delta.ne(600)])
        assert bad==[a['row'] for a in ANOMALIES]
        for a in ANOMALIES:
            assert str(dates.iloc[a['previous_row']])==a['previous']
            assert str(dates.iloc[a['row']])==a['current'] and delta.iloc[a['row']]==a['seconds']
        assert list(frame.index[dates.duplicated()])==[19044]
        assert list(frame.index[frame.duplicated()])==[19044]
        assert frame.iloc[19043].equals(frame.iloc[19044])
        assert not frame.isna().any().any() and not any(frame[c].nunique(dropna=False)==1 for c in frame)
    else:assert dates.is_unique and delta.dropna().eq(3600).all()
    numeric=frame.iloc[:,1:]
    assert all(pd.api.types.is_numeric_dtype(t) for t in numeric.dtypes)
    values=frame[spec['inputs']].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    train=spec['train'][1];offline=spec['online'][0]
    assert offline==len(values)//4 and train==offline*4//5
    scaler=StandardScaler().fit(values[:train]);scaled=scaler.transform(values)
    np.testing.assert_array_equal(scaler.mean_,values[:train].mean(axis=0))
    target=spec['inputs'].index(spec['target'])
    def windows(start,end):
        sequence=scaled[start:end]
        x=np.array([sequence[i:i+24] for i in range(len(sequence)-24)])
        y=np.array([sequence[i+24] for i in range(len(sequence)-24)])
        x=torch.tensor(x,dtype=torch.float32)
        y=torch.tensor(y[:,target:target+1],dtype=torch.float32)
        times=np.linspace(0,24,num=24)
        times=np.hstack([times[:,None]]*x.shape[0])
        t=torch.from_numpy(times[:,:,None]).to(torch.float32).transpose(0,1)
        return x,y,t
    data=dict(train=windows(0,train),validation=windows(train-24,offline),stream=windows(offline-24,len(values)))
    for key,expected in [('train',spec['train_windows']),('validation',spec['validation_windows']),('stream',spec['online_windows'])]:
        x,y,t=data[key]
        assert x.shape==(expected,24,spec['input_dim']) and y.shape==(expected,1) and t.shape==(expected,24,1)
    manifest=dict(spec,scaler=dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                                  var=scaler.var_.tolist(),n_samples_seen=int(scaler.n_samples_seen_)))
    return data,manifest


def batch(data,index,size=1):
    import numpy as np
    x,y,t=data;start=index*size
    x=x[start:start+size].transpose(0,1);t=t[start:start+size].transpose(0,1)
    if size==64:
        # Preserve historical gen_batch RNG consumption and temporal slice.
        n_sample=min(23,x.shape[0]);time_len=x.shape[0]
        t0=int(np.argmax(np.random.multinomial(1,[1./(time_len-n_sample)]*(time_len-n_sample))))
        return x[t0:t0+n_sample+1],t[t0:t0+n_sample+1],y[start:start+size]
    return x,t,y[start:start+size]
