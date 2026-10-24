"""Rebuild the blog's illustrative NF4 plot (requires numpy and matplotlib)."""
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'static/images/lora-qlora'
normal = NormalDist()
offset = 0.9677083
positive = [normal.inv_cdf(p) for p in np.linspace(0.5, offset, 9)[1:]]
negative = [-normal.inv_cdf(p) for p in np.linspace(0.5, offset, 8)[1:]]
codebook = np.sort(negative + [0.0] + positive)
codebook /= codebook[-1]
uniform = np.linspace(-1, 1, 16)

rng = np.random.default_rng(42)
weights = rng.normal(size=(4096, 64))
scales = np.max(np.abs(weights), axis=1, keepdims=True)
normalized = weights / scales

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False})
fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
fig.patch.set_facecolor('#f8fafc')
for ax, levels, title, color in zip(
    axes, (uniform, codebook), ('Uniform 4-bit: evenly spaced levels',
                              'NF4: more levels near zero'), ('#64748b', '#0f766e')
):
    codes = np.abs(normalized[..., None] - levels).argmin(axis=-1)
    reconstructed = levels[codes] * scales
    mse = np.mean((weights - reconstructed) ** 2)
    print(f'{title}: MSE={mse:.6f}')
    ax.set_facecolor('#f8fafc')
    ax.hist(normalized.ravel(), bins=100, range=(-1, 1), density=True,
            color='#cbd5e1', edgecolor='none', alpha=0.8)
    boundaries = (levels[:-1] + levels[1:]) / 2
    for boundary in boundaries:
        ax.axvline(boundary, color=color, alpha=0.45, linewidth=0.9, linestyle='--')
    ax.plot(levels, [-0.10] * 16, '|', color=color, markersize=16, markeredgewidth=2.5,
            clip_on=False)
    ax.set_ylim(-0.19, 1.6)
    ax.set_yticks([0, 0.5, 1, 1.5])
    ax.set_ylabel('Density')
    ax.set_title(f'{title}   |   MSE: {mse:.4f}', loc='left', color=color,
                 fontsize=14, weight='bold', pad=13)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.tick_params(colors='#475569')
axes[-1].set_xlim(-1.04, 1.04)
axes[-1].set_xlabel('Weight / block absolute maximum', labelpad=10)
fig.suptitle('Same weights. Same 16 codes. Different spacing.', x=0.09, y=0.98,
             ha='left', fontsize=19, weight='bold', color='#0f172a')
fig.text(0.09, 0.02, 'Synthetic N(0, 1) weights · 4,096 blocks of 64 · same absmax scales\n'
         'Gray: normalized weights   |   Solid ticks: codebook values   |   Dashed lines: bin boundaries\n'
         'MSE is measured after reconstruction, in original weight units.',
         color='#475569', fontsize=10, linespacing=1.7)
fig.subplots_adjust(top=0.88, bottom=0.22, hspace=0.50, left=0.09, right=0.98)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / 'nf4-vs-uniform.svg', facecolor=fig.get_facecolor())
fig.savefig('/tmp/nf4-vs-uniform.png', dpi=150, facecolor=fig.get_facecolor())
print('NF4:', [round(float(x), 4) for x in codebook])
