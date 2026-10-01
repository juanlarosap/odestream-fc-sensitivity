"""Frozen stochastic PaperLoss-B and numerical diagnostics."""
import torch
import numpy as np

def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b

def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999))

def paperloss(target, prediction, z, mu, log_var):
    """Authorized Interpretation B, shared by warm-up and online updates."""
    mse = (target - prediction).square().mean()
    kl = (-0.5 * (1 + log_var - mu.square() - log_var.exp()).sum(dim=-1)).mean()
    l1 = (0.01 * z.abs().sum(dim=-1)).mean()
    return dict(mse_loss=mse, kl_loss=kl, l1_loss=l1, total_loss=mse + kl + l1)

def finite(value, label):
    if isinstance(value, torch.Tensor):
        ok = bool(torch.isfinite(value).all())
    elif isinstance(value, dict):
        for k, v in value.items():
            finite(v, label + '/' + str(k))
        return
    elif isinstance(value, (list, tuple)):
        for v in value:
            finite(v, label)
        return
    elif isinstance(value, (float, int, np.number, np.ndarray)):
        ok = bool(np.isfinite(value).all())
    else:
        return
    if not ok:
        raise FloatingPointError('Non-finite ' + label)

def diagnostics(z, mu, log_var):
    with torch.no_grad():
        sigma = (0.5 * log_var).exp()
        finite({'z': z, 'mu': mu, 'log_var': log_var, 'sigma': sigma}, 'latent')
        return dict(mean_abs_mu=mu.abs().mean().item(), mean_sigma=sigma.mean().item(),
                    median_sigma=torch.quantile(sigma.flatten(), .5).item(),
                    max_sigma=sigma.max().item(), mean_abs_z=z.abs().mean().item())
