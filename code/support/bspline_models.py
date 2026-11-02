import numpy as np
import torch
from torch import nn
NAME = 'BSpline_quadratic_16_dreal'
KNOTS = [-2.0, -2.0, -2.0, 0.0, 2.0, 2.0, 2.0]
POLYNOMIALS = {'negative': [[0.0, 0.0, 0.25], [0.5, -0.5, -0.375], [0.5, 0.5, 0.125], [0.0, 0.0, 0.0]], 'positive': [[0.0, 0.0, 0.0], [0.5, -0.5, 0.125], [0.5, 0.5, -0.375], [0.0, 0.0, 0.25]]}
CONFIGS = {NAME: dict(n_params=16, degree=2, basis_per_axis=4, knots=KNOTS, knots_trainable=False, coefficient_shape=[4, 4], budget='small')}

def basis_torch(x):
    negative, positive = ([], [])
    for key, target in [('negative', negative), ('positive', positive)]:
        for a, b, c in POLYNOMIALS[key]:
            target.append(a + b * x + c * x * x)
    return torch.where((x <= 0)[:, None], torch.stack(negative, 1), torch.stack(positive, 1))

class QuadraticTensorBSpline(nn.Module):

    def __init__(self):
        super().__init__()
        self.coeff = nn.Parameter(0.05 * torch.randn(4, 4))

    def forward(self, x):
        b1, b2 = (basis_torch(x[:, 0]), basis_torch(x[:, 1]))
        return torch.einsum('bi,ij,bj->b', b1, self.coeff, b2).unsqueeze(1)

    def training_loss(self, x, dynamics, dataset_size):
        V = x.square().sum(1, keepdim=True) * self(x)
        gradient = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
        LV = (gradient * dynamics(x)).sum(1, keepdim=True)
        target = x.square().sum(1, keepdim=True).sqrt()
        return (torch.relu(-V) + 3.5 * torch.relu(LV + 0.5)).mean() + 5 * ((target - V) ** 2).mean()

    def snapshot(self):
        return dict(kind='exact_quadratic_tensor_BSpline', degree=2, knots=KNOTS, coeff=self.coeff.detach().cpu().double().tolist(), basis_polynomials=POLYNOMIALS, partition=0.0, knot_continuity='C1', extrapolation='extend first and last polynomial pieces; no zero truncation', symbolic_derivative='differentiate polynomial on each of four closed regions')

def build(name=NAME):
    assert name == NAME
    torch.manual_seed(1)
    model = QuadraticTensorBSpline()
    assert sum((param.numel() for param in model.parameters())) == 16
    return model

def numpy_raw(snapshot, x):
    x = np.asarray(x, dtype=np.float64).reshape(-1, 2)
    bases = []
    for axis in range(2):
        z = x[:, axis]
        negative, positive = [np.stack([a + b * z + c * z * z for a, b, c in POLYNOMIALS[key]], 1) for key in ['negative', 'positive']]
        bases.append(np.where((z <= 0)[:, None], negative, positive))
    return np.einsum('bi,ij,bj->b', bases[0], np.asarray(snapshot['coeff']), bases[1])
