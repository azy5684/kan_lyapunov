import math
import numpy as np
import torch
from torch import nn
CONFIGS = {'SVGP_16_dreal': dict(inducing_count=6, n_params=16, budget='small'), 'SVGP_126_dreal': dict(inducing_count=61, n_params=126, budget='KAN221_matched')}

def logit(p):
    return math.log(p / (1 - p))

class SparseVariationalGP(nn.Module):

    def __init__(self, inducing_count):
        super().__init__()
        self.inducing_count = inducing_count
        self.jitter = 1e-06
        self.register_buffer('inducing', torch.cat((torch.zeros(1, 2), 4 * torch.quasirandom.SobolEngine(2, scramble=True, seed=101).draw(inducing_count - 1) - 2)).double())
        self.raw_length = nn.Parameter(torch.full((2,), logit((1.5 - 0.15) / 3.85), dtype=torch.float64))
        self.raw_amplitude = nn.Parameter(torch.tensor(logit((1.0 - 0.1) / 3.9), dtype=torch.float64))
        self.raw_noise = nn.Parameter(torch.tensor(logit((1.0 - 0.03) / 2.97), dtype=torch.float64))
        self.q_mean = nn.Parameter(torch.zeros(inducing_count, dtype=torch.float64))
        self.q_log_std = nn.Parameter(torch.full((inducing_count,), math.log(0.1), dtype=torch.float64))
        with torch.no_grad():
            self.q_mean.copy_(torch.linalg.solve_triangular(self.cholesky(), torch.ones(inducing_count, 1, dtype=torch.float64), upper=False).ravel())

    def hyperparameters(self):
        return (0.15 + 3.85 * self.raw_length.sigmoid(), 0.1 + 3.9 * self.raw_amplitude.sigmoid(), 0.03 + 2.97 * self.raw_noise.sigmoid())

    def kernel(self, x, z):
        length, amplitude, _ = self.hyperparameters()
        diff = (x.double()[:, None, :] - z[None, :, :]) / length
        return amplitude.square() * torch.exp(-0.5 * diff.square().sum(-1))

    def cholesky(self):
        return torch.linalg.cholesky(self.kernel(self.inducing, self.inducing) + self.jitter * torch.eye(self.inducing_count, dtype=torch.float64))

    def posterior(self, x, variance=False):
        A = torch.linalg.solve_triangular(self.cholesky(), self.kernel(x, self.inducing).T, upper=False).T
        mean = A @ self.q_mean
        if not variance:
            return mean.unsqueeze(1)
        _, amplitude, _ = self.hyperparameters()
        var = amplitude.square() + (A.square() * (torch.exp(2 * self.q_log_std) - 1)).sum(1)
        if float(var.detach().min()) < -1e-07:
            raise FloatingPointError('Invalid posterior variance')
        return (mean.unsqueeze(1), var.clamp_min(0).unsqueeze(1))

    def forward(self, x):
        return self.posterior(x)

    def training_loss(self, x, f, dataset_size):
        mean, variance = self.posterior(x, variance=True)
        radius2 = x.square().sum(1, keepdim=True)
        V = radius2 * mean
        gradient = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
        LV = (gradient * f(x)).sum(1, keepdim=True)
        target = radius2.sqrt()
        _, _, noise = self.hyperparameters()
        expected_error2 = (target - V).square() + radius2.square() * variance
        gaussian_nll = 0.5 * (expected_error2 / noise.square() + 2 * noise.log() + math.log(2 * math.pi)).mean()
        KL = 0.5 * (torch.exp(2 * self.q_log_std) + self.q_mean.square() - 1 - 2 * self.q_log_std).sum()
        loss = (torch.relu(-V) + 3.5 * torch.relu(LV + 0.5)).mean() + 10 * (gaussian_nll + KL / dataset_size)
        return loss

    @torch.no_grad()
    def snapshot(self):
        length, amplitude, noise = self.hyperparameters()
        alpha = torch.linalg.solve_triangular(self.cholesky().T, self.q_mean[:, None], upper=True).ravel()
        return dict(kind='exact_frozen_GP_posterior_mean', inducing=self.inducing.tolist(), length=length.tolist(), amplitude=float(amplitude), noise=float(noise), alpha=alpha.tolist(), jitter=self.jitter, q_mean=self.q_mean.tolist(), q_std=self.q_log_std.exp().tolist(), uncertainty_smt_verified=False)

def build(name):
    torch.manual_seed(1)
    model = SparseVariationalGP(CONFIGS[name]['inducing_count'])
    assert sum((p.numel() for p in model.parameters() if p.requires_grad)) == CONFIGS[name]['n_params']
    return model

def numpy_mean(snapshot, x):
    x = np.asarray(x, dtype=float)
    z = np.asarray(snapshot['inducing'])
    delta = (x[:, None, :] - z[None, :, :]) / np.asarray(snapshot['length'])
    return snapshot['amplitude'] ** 2 * np.exp(-0.5 * (delta ** 2).sum(-1)) @ np.asarray(snapshot['alpha'])
