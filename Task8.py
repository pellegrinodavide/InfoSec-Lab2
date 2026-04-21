# Task 8: Implement the wiretap AWGN channel with PAM-2^lx modulation

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

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
        self.M    = 2 ** lx                         # 128
        raw       = pam_symbols(self.M)             # length M, values ±1 … ±127
        # Normalise so that E[x̃²] = 1  (unit average symbol power)
        self._scale     = np.sqrt(np.mean(raw ** 2))
        self.constellation = raw / self._scale      # unit-power symbols

    def modulate(self, x_int: np.ndarray) -> np.ndarray:
        """Map integer indices → real-valued PAM symbols."""
        return self.constellation[x_int]

    def demodulate(self, y_tilde: np.ndarray) -> np.ndarray:
        """
        Minimum-distance hard demodulation.
        Returns integer indices in {0, …, M-1}.
        """
        # y_tilde shape: (N,)  |  constellation shape: (M,)
        # Compute distances for all pairs and pick the closest symbol
        diff = y_tilde[:, np.newaxis] - self.constellation[np.newaxis, :]  # (N, M)
        return np.argmin(diff ** 2, axis=1)


# ── Wiretap AWGN channel ──────────────────────────────────────────────────────

class WiretapAWGN:
    """
    Wiretap AWGN channel with PAM-2^lx modulation.

    Parameters
    ----------
    snr_b_db : float
        Bob's SNR in dB.
    snr_e_db : float
        Eve's SNR in dB.
    lx : int
        Number of bits per symbol (default 7, so M = 128).
    rng : np.random.Generator, optional

    Channel model
    -------------
        ỹ = x̃ + N(0, P · 10^{-SNR_B/10})
        z̃ = x̃ + N(0, P · 10^{-SNR_E/10})

    With unit symbol power (P = 1) and our normalised constellation E[x̃²]=1.
    """

    def __init__(self, snr_b_db: float, snr_e_db: float,
                 lx: int = 7, rng=None):
        self.snr_b_db = snr_b_db
        self.snr_e_db = snr_e_db
        self.lx       = lx
        self.mod      = PAMModulator(lx)
        self.rng      = rng or np.random.default_rng(42)

        # Linear SNR values
        self._snr_b_lin = 10 ** (snr_b_db / 10.0)
        self._snr_e_lin = 10 ** (snr_e_db / 10.0)

        # Noise standard deviations (P = E[x̃²] = 1)
        self._sigma_b = np.sqrt(1.0 / self._snr_b_lin)
        self._sigma_e = np.sqrt(1.0 / self._snr_e_lin)

    # ── internal: pass a batch of real symbols through one AWGN + demod ──────
    def _awgn_demod(self, x_tilde: np.ndarray, sigma: float) -> np.ndarray:
        noise  = self.rng.normal(0.0, sigma, size=x_tilde.shape)
        y_tilde = x_tilde + noise
        return self.mod.demodulate(y_tilde)

    # ── public interface ──────────────────────────────────────────────────────
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

def ser_pam_theory(M: int, snr_lin: float) -> float:
    """
    Theoretical SER for PAM-M with normalised unit symbol power.

    SER = 2*(1 - 1/M) * Q( sqrt(3*SNR / (M²-1)) )
        = (1 - 1/M) * erfc( sqrt( 3*SNR / (2*(M²-1)) ) )

    This is exact for equally-spaced PAM with unit average power.
    """
    arg = np.sqrt(3.0 * snr_lin / (2.0 * (M ** 2 - 1)))
    return 2.0 * (1.0 - 1.0 / M) * 0.5 * erfc(arg)


def verify_channel(n_symbols: int = 2 ** 14):
    """
    Verify the AWGN channel implementation.

    Two checks are performed:
    1. Empirical SER_B and SER_E match the theoretical PAM-M SER formula.
    2. The ratio SER_B / SER_E (empirical) matches the ratio of the two
       theoretical SERs — this is the exact version of the handout's
       sqrt(SNR_B / SNR_E) approximation, valid for all SNR regimes.
    """

    test_pairs_db = [
        (10.0,  5.0),
        (15.0, 10.0),
        (20.0, 15.0),
        (25.0, 20.0),
        (30.0, 25.0),
    ]

    print("\nTask 8 — Wiretap AWGN channel verification")
    print(f"  M = 2^7 = 128 PAM symbols,  N = {n_symbols} symbols per experiment\n")
    header = (f"{'SNR_B':>7} {'SNR_E':>7} | "
              f"{'SER_B emp':>10} {'SER_B th':>10} | "
              f"{'SER_E emp':>10} {'SER_E th':>10} | "
              f"{'ratio emp':>10} {'ratio th':>10}")
    print(header)
    print("-" * len(header))

    snrb_list, snre_list = [], []
    ser_b_list, ser_e_list = [], []
    ser_b_th_list, ser_e_th_list = [], []
    ratio_emp_list, ratio_th_list = [], []

    for snrb_db, snre_db in test_pairs_db:
        ch = WiretapAWGN(snrb_db, snre_db)

        # Random symbols uniformly in {0, …, M-1}
        x_int = ch.rng.integers(0, ch.mod.M, size=n_symbols)
        y_int, z_int = ch.transmit(x_int)

        # Empirical SER
        ser_b = float(np.mean(x_int != y_int))
        ser_e = float(np.mean(x_int != z_int))

        # Theoretical SER using exact PAM formula
        snrb_lin = 10 ** (snrb_db / 10.0)
        snre_lin = 10 ** (snre_db / 10.0)
        ser_b_th = ser_pam_theory(ch.mod.M, snrb_lin)
        ser_e_th = ser_pam_theory(ch.mod.M, snre_lin)

        # Ratio: empirical vs theoretical (exact)
        ratio_emp = (ser_b / ser_e) if ser_e > 0 else float('inf')
        ratio_th  = (ser_b_th / ser_e_th) if ser_e_th > 0 else float('inf')

        snrb_list.append(snrb_db);      snre_list.append(snre_db)
        ser_b_list.append(ser_b);       ser_e_list.append(ser_e)
        ser_b_th_list.append(ser_b_th); ser_e_th_list.append(ser_e_th)
        ratio_emp_list.append(ratio_emp); ratio_th_list.append(ratio_th)

        print(f"{snrb_db:>7.1f} {snre_db:>7.1f} | "
              f"{ser_b:>10.5f} {ser_b_th:>10.5f} | "
              f"{ser_e:>10.5f} {ser_e_th:>10.5f} | "
              f"{ratio_emp:>10.4f} {ratio_th:>10.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Task 8 — Wiretap AWGN PAM-128 verification", fontsize=13, fontweight="bold")

    # Left: empirical SER vs theoretical SER for Bob and Eve
    ax = axes[0]
    snrb_arr     = np.array(snrb_list)
    snre_arr     = np.array(snre_list)
    sort_b = np.argsort(snrb_arr)
    sort_e = np.argsort(snre_arr)
    ax.semilogy(snrb_arr[sort_b], np.array(ser_b_list)[sort_b],
                'o-', color='steelblue', lw=2, ms=6, label="SER Bob (empirical)")
    ax.semilogy(snrb_arr[sort_b], np.array(ser_b_th_list)[sort_b],
                'o--', color='steelblue', lw=1.5, ms=4, alpha=0.6, label="SER Bob (theory)")
    ax.semilogy(snre_arr[sort_e], np.array(ser_e_list)[sort_e],
                's-', color='tomato', lw=2, ms=6, label="SER Eve (empirical)")
    ax.semilogy(snre_arr[sort_e], np.array(ser_e_th_list)[sort_e],
                's--', color='tomato', lw=1.5, ms=4, alpha=0.6, label="SER Eve (theory)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Symbol Error Rate")
    ax.set_title("Empirical vs Theoretical SER")
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)

    # Right: ratio SER_B/SER_E empirical vs theoretical (exact PAM formula)
    ax2 = axes[1]
    x_ax = np.arange(len(test_pairs_db))
    width = 0.35
    ax2.bar(x_ax - width/2, ratio_emp_list, width,
            label="Empirical  SER_B / SER_E", color='steelblue', alpha=0.8)
    ax2.bar(x_ax + width/2, ratio_th_list, width,
            label="Theoretical  SER_B / SER_E  (PAM formula)",
            color='darkorange', alpha=0.8)
    ax2.set_xticks(x_ax)
    ax2.set_xticklabels([f"({b:.0f},{e:.0f})" for b, e in test_pairs_db],
                        rotation=20, fontsize=9)
    ax2.set_xlabel("(SNR_B dB,  SNR_E dB)")
    ax2.set_ylabel("SER_B / SER_E")
    ax2.set_title("Ratio of SERs: empirical vs theoretical")
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
