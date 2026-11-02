# KAN Lyapunov

Run the following commands from the project root.

## 1. Install dependencies

```bash
pip install -r code/requirements.txt
```

## 2. Train a Lyapunov function

```bash
jupyter notebook code/train_kan_lyapunov.ipynb
```

## 3. Generate random systems

```bash
python code/generate_random_systems.py --output systems_generated
```

## 4. Load saved models

```bash
python code/load_saved_model.py models/mlp_2_2_1/cluster_1/system_001.pt
```

Method directories under `models/`: `kan_2_2_1`, `mlp_2_2_1`, `mlp_2_24_2_1`, `rbf`, `svgp`, and `b_spline`.

## 5. Plot success rate differences and ROA area comparisons

```bash
python code/plot_verification_statistics.py
```

## 6. Plot counterexample distances and ROA boundaries

Run each script in MATLAB:

```matlab
run('code/plot_counterexample_history.m')
run('code/plot_roa_comparison.m')
```

Figures are saved in `figures/`. Summary statistics are available in `results_summary.csv`.

Optional verification helper: `recheck_candidate` in `code/support/optional_verification.py`.
