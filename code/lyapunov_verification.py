import torch
from dreal import Expression, logical_and, logical_imply, logical_not, CheckSatisfiability

def CheckLyapunov(x, f, V, ball_lb, ball_ub, config, epsilon):
    ball = Expression(0)
    lie_derivative_of_V = Expression(0)
    for i in range(len(x)):
        ball += x[i] * x[i]
        lie_derivative_of_V += f[i] * V.Differentiate(x[i])
    ball_in_bound = logical_and(ball_lb * ball_lb <= ball, ball <= ball_ub * ball_ub)
    condition = logical_and(logical_imply(ball_in_bound, V > 0), logical_imply(ball_in_bound, lie_derivative_of_V < epsilon))
    return CheckSatisfiability(logical_not(condition), config)

def Tune(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).sum(dim=1, keepdim=True).sqrt()
