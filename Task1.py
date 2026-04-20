import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations
# Requirements: pip install numpy matplotlib
# ── helpers ───────────────────────────────────────────────────────────────────

def bits_to_int(b):
    return int(''.join(map(str, b)), 2)

def int_to_bits(v, n=7):
    return np.array([(v >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)

def ball(n, radius):
    """All n-bit integers with Hamming weight ≤ radius."""
    result = []
    for w in range(radius + 1):
        for positions in combinations(range(n), w):
            val = 0
            for p in positions:
                val |= (1 << (n - 1 - p))
            result.append(val)
    return np.array(result, dtype=np.int32)

def empirical_pmf(samples, support):
    counts = np.array([(samples == v).sum() for v in support], dtype=float)
    return counts / counts.sum()

# ── channel ───────────────────────────────────────────────────────────────────

class WiretapUniformErrorChannel:
    def __init__(self, n=7, r=1, s=3, rng=None):
        self.n   = n
        self.By  = ball(n, r)
        self.Bz  = ball(n, s)
        self.rng = rng or np.random.default_rng(42)
        print(f"|B_r={r}| = {len(self.By)},  |B_s={s}| = {len(self.Bz)}")

    def transmit(self, x_int, N):
        """Returns (y, z) arrays of N integer-valued outputs."""
        ey = self.rng.choice(self.By, size=N)
        ez = self.rng.choice(self.Bz, size=N)
        return x_int ^ ey, x_int ^ ez

# ── simulation ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N      = 2**14
    n, r, s = 7, 1, 3
    x_int  = bits_to_int([1,0,0,1,0,0,0])   # 1001000

    ch = WiretapUniformErrorChannel(n, r, s)
    y_samp, z_samp = ch.transmit(x_int, N)

    all_y = np.sort(ch.By ^ x_int)
    all_z = np.sort(ch.Bz ^ x_int)
    py_hat = empirical_pmf(y_samp, all_y)
    pz_hat = empirical_pmf(z_samp, all_z)

    # ── conditional independence: empirical I(y;z) ───────────────────────────────

    py_d = {int(v): p for v, p in zip(all_y, py_hat)}
    pz_d = {int(v): p for v, p in zip(all_z, pz_hat)}
    joint = {}
    for yv, zv in zip(y_samp, z_samp):
        k = (int(yv), int(zv))
        joint[k] = joint.get(k, 0) + 1

    MI = sum(
        (cnt/N) * np.log2((cnt/N) / (py_d[yv] * pz_d[zv]))
        for (yv, zv), cnt in joint.items()
        if py_d.get(yv, 0) > 0 and pz_d.get(zv, 0) > 0
    )
    print(f"Empirical I(y;z) = {MI:.6f} bits  (expected 0, residual is sampling noise)")

    # ── plots ─────────────────────────────────────────────────────────────────────

    label = lambda v: format(v, f'0{n}b')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Task 1 — Wiretap Uniform Error Channel  (x = 1001000, r=1, s=3)", fontsize=13)

    for ax, vals, pmf, ball_size, color, title in [
        (axes[0], all_y, py_hat, len(ch.By), 'steelblue',   r'Legitimate $\hat{p}_{y|x}$'),
        (axes[1], all_z, pz_hat, len(ch.Bz), 'darkorange',  r'Eavesdropper $\hat{p}_{z|x}$'),
    ]:
        step = max(1, len(vals) // 16)
        ax.bar(range(len(vals)), pmf, color=color, alpha=0.8, label='Empirical')
        ax.axhline(1/ball_size, color='crimson', lw=1.8, ls='--', label=f'Uniform 1/{ball_size}')
        ax.set_xticks(range(0, len(vals), step))
        ax.set_xticklabels([label(v) for v in vals[::step]], rotation=90, fontsize=7)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Output word')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    axes[1].text(0.98, 0.95, f'I(y;z) = {MI:.4f} bits\n(≈ 0 → independent)',
                transform=axes[1].transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.tight_layout()
    plt.savefig("Task1Image/task1_wiretap_uniform.png", dpi=150, bbox_inches='tight')
    plt.show()