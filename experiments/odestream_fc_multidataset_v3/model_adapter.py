"""Explicit model dependencies; preserves historical construction and RNG ordering."""
import copy
from . import models, runtime
from .losses import optimizer, paperloss, finite
from .fc import FunctionalConsistencyState, stochastic_forward, functional_consistency_training_step


def combined(dim,vae=None):
    model=models.CombinedModel(vae if vae is not None else models.ODEVAE(1,64,64,dim),
        models.LSTMModel(dim,64,2,1,24),models.ConcatenationLayer(),dim,1)
    return model.train()


class Model:
    def __init__(self,dim,variant,phase,coefficient=None,provenance=None,checkpoint=None):
        self.dim=dim;self.variant=variant;self.phase=phase
        self.coefficient=coefficient;self.provenance=provenance or {}
        self.model=models.ODEVAE(1,64,64,dim) if phase=='warmup' else combined(dim)
        self.model.train();self.opt=optimizer(self.model)
        self.method=self.new_method() if variant=='fc' else None
        if checkpoint is not None:self.restore(checkpoint)
        if any(p.device.type!='cpu' for p in self.model.parameters()):raise RuntimeError('CPU-only model required')
    def new_method(self):
        return FunctionalConsistencyState(self.model,self.coefficient,'rho',self.provenance)
    def snapshot(self):
        return copy.deepcopy(dict(model=self.model.state_dict(),optimizer=self.opt.state_dict(),
            method=self.method.state_dict() if self.method else None,rng=runtime.rng()))
    def restore(self,ck):
        self.model.load_state_dict(ck['model'],strict=True);self.opt.load_state_dict(copy.deepcopy(ck['optimizer']))
        if self.method:self.method.load_state_dict(ck['method'])
        elif ck['method'] is not None:raise ValueError('Unexpected FC state')
        runtime.restore_rng(ck['rng'])
    def forward(self,batch):
        x,t,y=batch
        if self.method:
            out,epsilon=stochastic_forward(self.model,x,t,None)
            return out,(out,epsilon)
        return self.model(x,t,None),None
    def update(self,batch,output=None,forward=None):
        if output is None:output,forward=self.forward(batch)
        if self.method:
            row,_=functional_consistency_training_step(self.method,self.opt,batch,student_forward=forward)
        else:
            self.opt.zero_grad();comp=paperloss(batch[2],output[0],*output[-3:]);finite(comp,'loss')
            comp['total_loss'].backward();finite([p.grad for p in self.model.parameters()],'gradient')
            row={k:v.item() for k,v in comp.items()};self.opt.step()
        finite(self.model.state_dict(),'model');finite(self.opt.state_dict(),'Adam');finite(row,'update')
        return row


def online_boundary(model):
    # Caller already loaded best VAE weights, leaving final validation RNG intact.
    network=combined(model.dim,model.model)
    weights=copy.deepcopy(network.state_dict());rng=runtime.rng()
    adapter=Model(model.dim,model.variant,'online',model.coefficient,model.provenance)
    adapter.model.load_state_dict(weights)
    if adapter.method:adapter.method=adapter.new_method()
    runtime.restore_rng(rng)
    return adapter.snapshot()
