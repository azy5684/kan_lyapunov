import argparse
from collections import Counter
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TIE_TOLERANCE = 0.01

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def font_setup():
    for folder in [Path('C:/Windows/Fonts'), Path('/mnt/c/Windows/Fonts')]:
        for name in ['arial.ttf', 'times.ttf']:
            path = folder / name
            if path.exists():
                font_manager.fontManager.addfont(str(path))
    names = {f.name for f in font_manager.fontManager.ttflist}
    sans = 'Arial' if 'Arial' in names else 'DejaVu Sans'
    serif = 'Times New Roman' if 'Times New Roman' in names else 'DejaVu Serif'
    plt.rcParams.update({'font.family': sans, 'font.size': 10.5, 'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10.5, 'axes.linewidth': 0.8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white', 'axes.unicode_minus': False})
    return serif

def load_success_differences():
    rows = read(ROOT / 'results_summary.csv')
    lookup = {(int(r['cluster']), r['method']): r for r in rows}
    methods = ['MLP_2-2-1', 'MLP_2-24-2-1', 'RBF', 'SVGP']
    values = []
    for method in methods:
        line = []
        for cluster in [1, 2, 3]:
            a, b = (lookup[cluster, 'KAN_2-2-1'], lookup[cluster, method])
            assert int(a['recorded']) == int(b['recorded']) == 100
            line.append(100 * (int(a['successes']) / int(a['recorded']) - int(b['successes']) / int(b['recorded'])))
        values.append(line)
    values = np.round(np.asarray(values), 10)
    assert np.allclose(values, [[12, 7, 21], [9, 16, 3], [1, 1, -1], [5, 5, 18]])
    return values

def load_area_counts():
    rows = read(HERE / 'kan_rbf_area_pairs.csv')
    assert len(rows) == 193
    assert len({(r['cluster'], r['system_id']) for r in rows}) == 193
    counts = []
    for cluster in ['1', '2', '3', None]:
        selected = [r for r in rows if cluster is None or r['cluster'] == cluster]
        tally = Counter()
        for r in selected:
            assert float(r['common_radius']) in [2.0, 0.5, 0.1]
            a, b = (float(r['area_A']), float(r['area_B']))
            assert a >= 0 and b >= 0 and np.isfinite([a, b]).all()
            scale = max(a, b)
            category = 'Similar' if scale == 0 or abs(a - b) / scale <= TIE_TOLERANCE else 'KAN larger' if a > b else 'RBF larger'
            assert r['winner'] == {'Similar': 'tie', 'KAN larger': 'A', 'RBF larger': 'B'}[category]
            tally[category] += 1
        counts.append([tally[k] for k in ['KAN larger', 'Similar', 'RBF larger']])
    counts = np.asarray(counts)
    assert np.array_equal(counts, [[73, 7, 9], [55, 10, 8], [17, 8, 6], [145, 25, 23]])
    return counts

def save(fig, out, stem, formats):
    for ext in formats:
        fig.savefig(out / f'{stem}.{ext}', dpi=300, bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)

def plot_success(values, out, formats):
    colors = ['#B9DDF6', '#FFDEA6', '#FADFE3']
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    fig.subplots_adjust(left=0.14, right=0.95, top=0.83, bottom=0.21)
    y = np.arange(4)
    height = 0.23
    for i in range(3):
        pos = y + (i - 1) * 0.27
        ax.barh(pos, values[:, i], height=height, color=colors[i], edgecolor='#444444', linewidth=0.65, label=f'Cluster {i + 1}', zorder=3)
        for py, value in zip(pos, values[:, i]):
            ax.text(value + (0.5 if value >= 0 else -0.45), py, f'{value:+.0f}', ha='left' if value >= 0 else 'right', va='center', fontsize=10.5)
    ax.set_yticks(y)
    ax.set_yticklabels(['MLP$_1$', 'MLP$_2$', 'RBF', 'SVGP'])
    ax.invert_yaxis()
    ax.set_xlim(-5, 25)
    ticks = np.arange(-5, 26, 5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'+{v}' if v > 0 else str(v) for v in ticks])
    ax.set_xlabel('KAN minus baseline (percentage points)')
    ax.grid(axis='x', color='#E4E4E4', linewidth=0.7, zorder=0)
    ax.axvline(0, color='black', linewidth=0.9, zorder=4)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=10)
    ax.legend(ncol=3, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.015), columnspacing=3, handlelength=2)
    save(fig, out, 'fig3_success_rate_difference', formats)

def plot_areas(counts, out, formats, serif):
    colors = ['#AEC6E7', '#E1E1E1', '#FFDEA0']
    fig = plt.figure(figsize=(7.5, 3.15))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.38], left=0.025, right=0.99, bottom=0.12, top=0.91, wspace=0.28)
    pie = fig.add_subplot(grid[0, 0])
    bars = fig.add_subplot(grid[0, 1])
    total = counts[-1]
    index = [0]

    def label(pct):
        n = int(total[index[0]])
        index[0] += 1
        return f'{pct:.1f}%\n(n = {n})'
    pie.pie(total, colors=colors, startangle=90, autopct=label, pctdistance=0.69, wedgeprops={'linewidth': 1, 'edgecolor': 'white'}, textprops={'fontsize': 10.5, 'fontfamily': serif})
    pie.set_aspect('equal')
    percentages = 100 * counts / counts.sum(axis=1)[:, None]
    left = np.zeros(4)
    for i in range(3):
        bars.barh(np.arange(4), percentages[:, i], left=left, height=0.56, color=colors[i], edgecolor='white', linewidth=0.8)
        for j in range(4):
            bars.text(left[j] + percentages[j, i] / 2, j, str(counts[j, i]), ha='center', va='center', fontsize=10.5, fontfamily=serif)
        left += percentages[:, i]
    bars.set_yticks(np.arange(4))
    bars.set_yticklabels([f'Cluster {i + 1}\n(n = {counts[i].sum()})' if i < 3 else f'Overall\n(n = {counts[i].sum()})' for i in range(4)], fontfamily=serif)
    bars.invert_yaxis()
    bars.set_xlim(0, 100)
    bars.set_xticks([0, 25, 50, 75, 100])
    bars.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    bars.tick_params(axis='both', length=0)
    bars.tick_params(axis='y', pad=8)
    for spine in bars.spines.values():
        spine.set_visible(False)
    fig.legend(handles=[Patch(facecolor=c, edgecolor='white', label=l) for c, l in zip(colors, ['KAN larger', 'Similar', 'RBF larger'])], ncol=3, frameon=False, loc='upper right', bbox_to_anchor=(0.995, 1.04), handlelength=1.2, handletextpad=0.4, columnspacing=2.1, prop={'family': serif, 'size': 10.5})
    save(fig, out, 'fig4_roa_area_comparison', formats)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'figures')
    parser.add_argument('--formats', nargs='+', choices=['png', 'pdf', 'svg'], default=['png'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    serif = font_setup()
    diff = load_success_differences()
    areas = load_area_counts()
    plot_success(diff, args.output, args.formats)
    plot_areas(areas, args.output, args.formats, serif)
    print('Fig. 3 differences:', diff.astype(int).tolist())
    print('Fig. 4 counts [KAN larger, Similar, RBF larger]:', areas.tolist())
    print('Output:', args.output.resolve())
if __name__ == '__main__':
    main()
