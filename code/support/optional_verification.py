from lyapunov_verification import CheckLyapunov

def recheck_candidate(variables, dynamics, candidate, ball_lb, ball_ub, config, epsilon=-1e-05):
    return CheckLyapunov(variables, dynamics, candidate, ball_lb, ball_ub, config, epsilon)
