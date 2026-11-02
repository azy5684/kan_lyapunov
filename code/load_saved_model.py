import copy
from pathlib import Path
import torch
from torch import nn
from kan import KAN
from support.svgp_models import SparseVariationalGP
from support.bspline_models import QuadraticTensorBSpline

class MLP(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(width[:-1], width[1:])])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return x

class RBF(nn.Module):

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.register_buffer('centers', torch.zeros(16, 2))
        self.linear = nn.Linear(16, 1)

    def forward(self, x):
        distance = (x[:, None, :] - self.centers[None, :, :]).square().sum(-1)
        return self.linear(1 / (1 + self.beta * distance))

def load_saved_model(path):
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=False)
    key = checkpoint['public_method_key']
    meta = checkpoint['architecture_metadata']
    if key == 'KAN_2-2-1':
        kw = meta.get('KAN_221_KWARGS') or meta.get('KAN_KWARGS')
        model = KAN(**copy.deepcopy(kw), auto_save=False)
    elif key.startswith('MLP_'):
        model = MLP(checkpoint['actual_architecture'])
    elif key == 'RBF':
        model = RBF(meta.get('RBF_BETA') or 1.0)
    elif key == 'SVGP':
        model = SparseVariationalGP(checkpoint['model_state_dict']['q_mean'].numel())
    elif key == 'B-Spline':
        model = QuadraticTensorBSpline()
    else:
        raise ValueError(key)
    dtype = next((v.dtype for v in checkpoint['model_state_dict'].values() if v.is_floating_point()))
    nn.Module.to(model, dtype=dtype)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return (model, checkpoint)

def lyapunov_value(model, x):
    return x.square().sum(1, keepdim=True) * model(x)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    model, checkpoint = load_saved_model(args.checkpoint)
    dtype = next(model.parameters()).dtype
    x = torch.tensor([[0.0, 0.0], [0.1, 0.1]], dtype=dtype)
    with torch.no_grad():
        print(checkpoint['display_name'], lyapunov_value(model, x).ravel().tolist())
