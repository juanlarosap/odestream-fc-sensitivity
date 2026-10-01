"""CPU scientific runtime helpers; no configuration changes at import."""
import math
import random
import numpy as np
import torch


def configure():
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    torch.use_deterministic_algorithms(True)
    if torch.get_num_threads()!=8 or torch.get_num_interop_threads()!=8:raise RuntimeError('CPU thread contract')


def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())
def restore_rng(value):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch'])
def seed(value):random.seed(value);np.random.seed(value);torch.manual_seed(value)
def norm(values):return math.sqrt(math.fsum(v.detach().double().square().sum().item() for v in values if v is not None))
def stats(values):
    v=np.asarray(values,dtype=float)
    if not len(v):return None
    return dict(n=len(v),min=float(v.min()),max=float(v.max()),mean=float(v.mean()),median=float(np.median(v)),
                **{f'Q{q}':float(np.quantile(v,q/100,method='linear')) for q in (25,50,75,90,95)})
