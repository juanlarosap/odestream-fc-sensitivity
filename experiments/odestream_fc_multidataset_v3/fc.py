"""Canonical same-epsilon FC; extracted historical update ordering."""
import copy
import math
import torch
from . import models
from .losses import paperloss, finite
from .runtime import norm
BETA = 0.99
EPS = 1e-12

def stochastic_forward(model, x, t, y=None, epsilon=None):
    """Original operation order; expose the one student draw or reuse a supplied tensor.

    Returns (historical output tuple, epsilon). Does not modify historical modules.
    Supports original VAE warm-up and original CombinedModel adaptation.
    """
    is_combined = isinstance(model,models.CombinedModel)
    vae = model.model1 if is_combined else model
    if not isinstance(vae,models.ODEVAE): raise TypeError('Expected historical ODEVAE/CombinedModel')
    mu,lv = vae.encoder(x,t)
    if epsilon is None:
        epsilon = torch.randn_like(mu)
    elif epsilon.shape != mu.shape or epsilon.dtype != mu.dtype or epsilon.device != mu.device:
        raise ValueError('Provided epsilon must exactly match latent shape/dtype/device')
    z = mu + epsilon * torch.exp(0.5*lv)  # Identical historical multiplication order.
    p1 = vae.decoder(z,t)
    if is_combined:
        p2 = model.model2(x)
        pred = model.final_layer(model.concatenation_layer(p1,p2))
        return (pred,p1,p2,z,mu,lv),epsilon
    return (p1,z,mu,lv),epsilon

@torch.no_grad()
def update_ema_teacher(teacher, student, beta=BETA):
    if beta != BETA: raise ValueError('EMA beta is fixed at 0.99')
    tp,sp=dict(teacher.named_parameters()),dict(student.named_parameters())
    if tp.keys()!=sp.keys(): raise ValueError('Teacher/student architecture mismatch')
    if list(student.buffers()) or list(teacher.buffers()):
        raise ValueError('Historical architecture has no buffers; new buffer semantics need review')
    before=[p.detach().clone() for p in tp.values()]
    for name,p in tp.items():
        p.copy_(beta*p+(1-beta)*sp[name])
    return norm([p.double()-b.double() for p,b in zip(tp.values(),before)])

class FunctionalConsistencyState:
    def __init__(self, student, lambda_consistency, level='disabled', provenance=None):
        if not math.isfinite(lambda_consistency) or lambda_consistency<0: raise ValueError('Invalid lambda')
        self.student=student
        self.teacher=copy.deepcopy(student)
        self.teacher.requires_grad_(False)
        for p in self.teacher.parameters(): p.grad=None
        if list(student.buffers()): raise ValueError('Unexpected model buffers')
        self.beta=BETA
        self.lambda_consistency=float(lambda_consistency)
        self.level=level
        self.provenance=copy.deepcopy(provenance or {})
        self.rows=[]
        self.student_update_count=0
        self.teacher_update_count=0

    def state_dict(self):
        return copy.deepcopy(dict(version=1,student=self.student.state_dict(),teacher=self.teacher.state_dict(),
            beta=self.beta,lambda_consistency=self.lambda_consistency,level=self.level,provenance=self.provenance,
            student_update_count=self.student_update_count,teacher_update_count=self.teacher_update_count,
            rows=self.rows,training=self.student.training))

    def load_state_dict(self,value):
        for key in ('version','beta','lambda_consistency','level','provenance'):
            expected=1 if key=='version' else getattr(self,key)
            if value[key]!=expected: raise ValueError('FC checkpoint mismatch: '+key)
        if value['student_update_count']!=value['teacher_update_count'] or value['student_update_count']!=len(value['rows']):
            raise ValueError('FC counters/history mismatch')
        self.student.load_state_dict(value['student'],strict=True)
        self.teacher.load_state_dict(value['teacher'],strict=True)
        self.student.train(value['training']); self.teacher.train(value['training'])
        self.student_update_count=value['student_update_count']; self.teacher_update_count=value['teacher_update_count']
        self.rows=copy.deepcopy(value['rows'])

def gradient_diagnostics(base,consistency,params):
    # autograd.grad never accumulates into .grad; retain for actual student backward.
    gb=torch.autograd.grad(base,params,retain_graph=True,allow_unused=True)
    gc=torch.autograd.grad(consistency,params,retain_graph=True,allow_unused=True)
    b,c=norm(gb),norm(gc)
    dot=math.fsum((x.detach().double()*y.detach().double()).sum().item()
                 for x,y in zip(gb,gc) if x is not None and y is not None)
    return dict(G_base=b,G_cons=c,unscaled_gradient_ratio=c/(b+EPS),
                gradient_cosine=dot/(b*c) if b>0 and c>0 else None)

def functional_consistency_training_step(state,optimizer,batch,gradient_metrics=False,student_forward=None):
    """One student Adam commit followed by one EMA commit.

    Optional student_forward=(output,epsilon) reuses a prior prediction/gating
    forward. SKIP simply does not call this function. Returns (row,prediction).
    """
    student,teacher=state.student,state.teacher
    if type(optimizer) is not torch.optim.Adam: raise TypeError('Requires historical Adam')
    p=[p for p in student.parameters() if p.requires_grad]
    optp=[q for g in optimizer.param_groups for q in g['params']]
    if [id(q) for q in optp]!=[id(q) for q in p]: raise ValueError('Adam must contain only all student parameters')
    if any(g['weight_decay']!=0 for g in optimizer.param_groups): raise ValueError('No weight decay')
    teacher.train(student.training)
    optimizer.zero_grad()
    x,t,y=batch
    output,epsilon=stochastic_forward(student,x,t,y) if student_forward is None else student_forward
    with torch.no_grad():
        teacher_output,teacher_epsilon=stochastic_forward(teacher,x,t,y,epsilon=epsilon)
    if teacher_epsilon is not epsilon: raise AssertionError('Teacher did not reuse exact epsilon tensor')
    components=paperloss(y,output[0],*output[-3:])
    base=components['total_loss']
    consistency=(output[0]-teacher_output[0]).square().mean()
    weighted=state.lambda_consistency*consistency
    # A true bypass guarantees baseline graph/backward identity in calibration.
    total=base if state.lambda_consistency==0 else base+weighted
    finite(dict(base=base,consistency=consistency,total=total),'loss')
    params=list(student.parameters())
    metrics=gradient_diagnostics(base,consistency,params) if gradient_metrics else {}
    if gradient_metrics:
        if any(q.grad is not None for q in params):
            # Historical Adam zero_grad may leave existing zero tensors.
            if any(q.grad is not None and torch.count_nonzero(q.grad) for q in params):
                raise AssertionError('Diagnostic gradients contaminated .grad')
        metrics['scaled_gradient_ratio']=state.lambda_consistency*metrics['unscaled_gradient_ratio']
    distance=norm([a.detach().double()-b.detach().double() for a,b in zip(student.parameters(),teacher.parameters())])
    row=dict(base_loss=base.item(),mse_loss=components['mse_loss'].item(),kl_loss=components['kl_loss'].item(),
             l1_loss=components['l1_loss'].item(),raw_consistency_loss=consistency.item(),
             weighted_consistency_loss=weighted.item(),total_loss=total.item(),lambda_consistency=state.lambda_consistency,
             teacher_student_prediction_abs_diff=(output[0]-teacher_output[0]).abs().mean().item(),
             teacher_student_prediction_squared_diff=consistency.item(),teacher_student_parameter_l2=distance,
             teacher_student_relative_parameter_l2=distance/(norm(student.parameters())+EPS),**metrics)
    total.backward()
    finite([q.grad for q in params if q.grad is not None],'student gradients')
    if any(q.requires_grad or q.grad is not None for q in teacher.parameters()): raise AssertionError('Teacher gradients')
    optimizer.step()
    state.student_update_count+=1
    row['EMA_update_norm']=update_ema_teacher(teacher,student,state.beta)
    state.teacher_update_count+=1
    row.update(student_update_count=state.student_update_count,teacher_update_count=state.teacher_update_count)
    finite(student.state_dict(),'student'); finite(teacher.state_dict(),'teacher'); finite(optimizer.state_dict(),'Adam')
    finite(row,'diagnostics')
    state.rows.append(row)
    return copy.deepcopy(row),output[0].detach().clone()
