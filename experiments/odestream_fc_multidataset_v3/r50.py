"""Prior-error rolling median gate; zero extra label delay; accepted ties."""
from collections import deque
import numpy as np


class Gate:
    def __init__(self,mode,errors=()):
        if mode not in ('always','r50'):raise ValueError('Unknown gate')
        self.window=deque(errors,maxlen=256) if mode=='r50' else None
        if self.window is not None and len(self.window)!=256:raise ValueError('Expected 256 bootstrap errors')
    def threshold(self):
        return float(np.quantile(self.window,.5,method='linear')) if self.window is not None else None
    def decide(self,error,tau):return int(tau is None or error>=tau)
    def append(self,error):
        if self.window is not None:self.window.append(error)
    def state(self):return list(self.window) if self.window is not None else None
