import argparse
import json
import random
from pathlib import Path
import numpy as np
BASE_SEED = 20260114
CLUSTERS = {1: dict(lambda_min=0.1, lambda_max=3.0, alpha=1.0), 2: dict(lambda_min=0.1, lambda_max=3.0, alpha=2.0), 3: dict(lambda_min=0.1, lambda_max=0.8, alpha=2.0)}
BASIS = ['x1^2', 'x2^2', 'x1*x2', 'x1^3', 'x2^3', 'x1^2*x2', 'x1*x2^2', 'sin(x1)', 'sin(x2)', 'sin(x1+x2)', 'cos(x1)-1', 'cos(x2)-1']

def generate(cluster):
    cfg = CLUSTERS[int(cluster)]
    systems = []
    for sid in range(100):
        seed = BASE_SEED + sid
        random.seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        theta = random.uniform(0, 2 * np.pi)
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        eigenvalues = [-random.uniform(cfg['lambda_min'], cfg['lambda_max']) for _ in range(2)]
        A = rotation @ np.diag(eigenvalues) @ rotation.T
        k1, k2 = (random.randint(1, 4), random.randint(1, 4))
        t1, t2 = (random.sample(BASIS, k1), random.sample(BASIS, k2))
        c1 = np.random.uniform(-cfg['alpha'], cfg['alpha'], k1)
        c2 = np.random.uniform(-cfg['alpha'], cfg['alpha'], k2)
        systems.append(dict(A=A.tolist(), terms_f1=t1, coeffs_f1=c1.tolist(), terms_f2=t2, coeffs_f2=c2.tolist(), generation_seed=seed))
    return systems

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cluster', choices=['1', '2', '3', 'all'], default='all')
    parser.add_argument('--output', type=Path, default=Path('systems_generated'))
    parser.add_argument('--check', action='store_true', help='Compare with bundled frozen system definitions')
    args = parser.parse_args()
    clusters = [1, 2, 3] if args.cluster == 'all' else [int(args.cluster)]
    for cluster in clusters:
        data = generate(cluster)
        if args.check:
            saved = Path(__file__).resolve().parent.parent / 'systems' / f'cluster_{cluster}.json'
            assert data == json.loads(saved.read_text(encoding='utf-8')), f'Cluster {cluster} mismatch'
            print(f'Cluster {cluster}: all 100 systems exactly match the saved definitions.')
        else:
            args.output.mkdir(parents=True, exist_ok=True)
            target = args.output / f'cluster_{cluster}.json'
            if target.exists():
                raise FileExistsError(f'Refusing to overwrite {target}')
            target.write_text(json.dumps(data, indent=2), encoding='utf-8')
            print(target)
if __name__ == '__main__':
    main()
