# Task 8: Implement the wiretap AWGN channel with PAM-2^lx modulation

import numpy as np
import matplotlib.pyplot as plt

# ── PAM-M modulator / demodulator ─────────────────────────────────────────────

def pam_symbols(M: int) -> np.ndarray:
    """
    Returns the M constellation points for a PAM-M modulation.
    Symbols are ±1, ±3, …, ±(M-1) (normalised later by the modulator).
    """
    return np.arange(-(M - 1), M, 2, dtype=float)   # [-M+1, -M+3, …, M-1]


class PAMModulator:
    """
    PAM-M modulator / demodulator for an alphabet of M = 2^lx symbols.

    The input alphabet is {0,1}^7  (7-bit integers), which we treat as a
    single symbol index in {0, …, M-1} where M = 2^7 = 128.

    Modulation:
        x̃ = constellation[x_int]            (real-valued)
    Demodulation (hard decision):
        x̂_int = argmin_k  |ỹ − constellation[k]|
    """

    def __init__(self, lx: int = 7):
        self.lx   = lx
        self.M    = 2 ** lx                          # 128
        raw       = pam_symbols(self.M)              # length M, values ±1 … ±127
        # Normalise so that E[x̃²] = 1  (unit average symbol power)
        self._scale    = np.sqrt(np.mean(raw ** 2))
        self.constellation = raw / self._scale       # unit-power symbols

    def modulate(self, x_int: np.ndarray) -> np.ndarray:
        """Map integer indices → real-valued PAM symbols."""
        return self.constellation[x_int]

    def demodulate(self, y_tilde: np.ndarray) -> np.ndarray:
        """
        Minimum-distance hard demodulation.
        Returns integer indices in {0, …, M-1}.
        """
        diff = y_tilde[:, np.newaxis] - self.constellation[np.newaxis, :]  # (N, M)
        return np.argmin(diff ** 2, axis=1)


# ── Wiretap AWGN channel ──────────────────────────────────────────────────────

class WiretapAWGN:
    """
    Wiretap AWGN channel with PAM-2^lx modulation.

    Channel model:
        ỹ = x̃ + N(0, P · 10^{-SNR_B/10})
        z̃ = x̃ + N(0, P · 10^{-SNR_E/10})

    With unit symbol power (P = E[x̃²] = 1).
    """

    def __init__(self, snr_b_db: float, snr_e_db: float,
                 lx: int = 7, rng=None):
        self.snr_b_db = snr_b_db
        self.snr_e_db = snr_e_db
        self.lx       = lx
        self.mod      = PAMModulator(lx)
        self.rng      = rng or np.random.default_rng(42)

        # Linear SNR → noise std  (P = 1)
        self._snr_b_lin = 10 ** (snr_b_db / 10.0)
        self._snr_e_lin = 10 ** (snr_e_db / 10.0)
        self._sigma_b   = np.sqrt(1.0 / self._snr_b_lin)
        self._sigma_e   = np.sqrt(1.0 / self._snr_e_lin)

    def _awgn_demod(self, x_tilde: np.ndarray, sigma: float) -> np.ndarray:
        noise   = self.rng.normal(0.0, sigma, size=x_tilde.shape)
        y_tilde = x_tilde + noise
        return self.mod.demodulate(y_tilde)

    def transmit(self, x_int: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Transmit a batch of symbol indices through Bob's and Eve's channels.

        Parameters
        ----------
        x_int : np.ndarray of int, shape (N,)
            Input symbol indices in {0, …, M-1}.

        Returns
        -------
        y_int, z_int : np.ndarray of int, shape (N,)
            Decoded symbol indices at Bob's and Eve's receivers.
        """
        x_tilde = self.mod.modulate(x_int)
        y_int   = self._awgn_demod(x_tilde, self._sigma_b)
        z_int   = self._awgn_demod(x_tilde, self._sigma_e)
        return y_int, z_int


# ── Verification ──────────────────────────────────────────────────────────────

def verify_channel(n_symbols: int = 2 ** 14):
    """
    Verify the AWGN channel implementation.

    The handout asks to compare:
        ratio of average symbol errors in the two outputs
                        vs
        square root of the ratio of the two linear SNRs

    i.e.   SER_E / SER_B  ≈  sqrt(SNR_B_lin / SNR_E_lin)

    Intuition: Eve has a noisier channel (lower SNR) so she makes more
    errors. The ratio SER_E / SER_B > 1 and grows with the SNR gap.
    At high SNR, where the nearest-neighbour approximation holds:
        SER ≈ 2(1-1/M) · Q(d_min / 2σ),  σ = 1/sqrt(SNR)
    so the ratio of SERs tends to sqrt(SNR_B / SNR_E).
    """

    # Gap of 10 dB → sqrt(SNR_B/SNR_E) = sqrt(10) ≈ 3.162 for all pairs
    test_pairs_db = [
        (10.0,  0.0),
        (15.0,  5.0),
        (20.0, 10.0),
        (25.0, 15.0),
        (30.0, 20.0),
    ]

    print("\nTask 8 — Wiretap AWGN channel verification")
    print(f"  M = 2^7 = 128 PAM symbols,  N = {n_symbols} symbols per experiment\n")
    header = (f"{'SNR_B (dB)':>11} {'SNR_E (dB)':>11} | "
              f"{'SER_B':>9} {'SER_E':>9} | "
              f"{'SER_E/SER_B':>12} {'√(SNR_B/SNR_E)':>15}")
    print(header)
    print("-" * len(header))

    snrb_list, snre_list = [], []
    ser_b_list, ser_e_list = [], []
    ratio_emp_list, ratio_th_list = [], []

    for snrb_db, snre_db in test_pairs_db:
        ch = WiretapAWGN(snrb_db, snre_db)

        # Random symbols uniformly in {0, …, M-1}
        x_int        = ch.rng.integers(0, ch.mod.M, size=n_symbols)
        y_int, z_int = ch.transmit(x_int)

        ser_b = float(np.mean(x_int != y_int))
        ser_e = float(np.mean(x_int != z_int))

        snrb_lin = 10 ** (snrb_db / 10.0)
        snre_lin = 10 ** (snre_db / 10.0)

        # Handout: ratio of errors  vs  sqrt(SNR_B / SNR_E)
        ratio_emp = (ser_e / ser_b)               if ser_b > 0 else float('inf')
        ratio_th  = np.sqrt(snrb_lin / snre_lin)  # theoretical approximation

        snrb_list.append(snrb_db);        snre_list.append(snre_db)
        ser_b_list.append(ser_b);         ser_e_list.append(ser_e)
        ratio_emp_list.append(ratio_emp); ratio_th_list.append(ratio_th)

        print(f"{snrb_db:>11.1f} {snre_db:>11.1f} | "
              f"{ser_b:>9.5f} {ser_e:>9.5f} | "
              f"{ratio_emp:>12.4f} {ratio_th:>15.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Task 8 — Wiretap AWGN PAM-128 verification",
                 fontsize=13, fontweight="bold")

    # Left: SER vs SNR for Bob and Eve
    ax       = axes[0]
    snrb_arr = np.array(snrb_list)
    snre_arr = np.array(snre_list)
    sort_b   = np.argsort(snrb_arr)
    sort_e   = np.argsort(snre_arr)
    ax.semilogy(snrb_arr[sort_b], np.array(ser_b_list)[sort_b],
                'o-', color='steelblue', lw=2, ms=6, label="SER Bob (legitimate)")
    ax.semilogy(snre_arr[sort_e], np.array(ser_e_list)[sort_e],
                's-', color='tomato',    lw=2, ms=6, label="SER Eve (eavesdropper)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Symbol Error Rate")
    ax.set_title("SER vs SNR")
    ax.legend(); ax.grid(True, which='both', alpha=0.3)

    # Right: SER_E/SER_B empirical vs sqrt(SNR_B/SNR_E) theoretical
    ax2   = axes[1]
    x_ax  = np.arange(len(test_pairs_db))
    width = 0.35
    ax2.bar(x_ax - width/2, ratio_emp_list, width,
            label=r"Empirical  $\mathrm{SER}_E / \mathrm{SER}_B$",
            color='steelblue', alpha=0.8)
    ax2.bar(x_ax + width/2, ratio_th_list, width,
            label=r"$\sqrt{\mathrm{SNR}_B^{lin} / \mathrm{SNR}_E^{lin}}$  (theory)",
            color='darkorange', alpha=0.8)
    ax2.set_xticks(x_ax)
    ax2.set_xticklabels([f"({b:.0f},{e:.0f})" for b, e in test_pairs_db],
                        rotation=20, fontsize=9)
    ax2.set_xlabel(r"(SNR$_B$ dB,  SNR$_E$ dB)")
    ax2.set_ylabel(r"$\mathrm{SER}_E / \mathrm{SER}_B$")
    ax2.set_title(r"Error ratio: empirical vs $\sqrt{\mathrm{SNR}_B/\mathrm{SNR}_E}$")
    ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("Task8Image/task8_awgn_verification.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved to Task8Image/task8_awgn_verification.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs("Task8Image", exist_ok=True)
    np.random.seed(42)
    verify_channel()
