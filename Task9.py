# Task 9: Evaluate the system security over the wiretap AWGN channel

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from Task2 import RandomBinningEncoder, CODEBOOK
from Task3 import RandomBinningDecoder
from Task8 import WiretapAWGN


# ── Information-theoretic helpers (same as Task 7) ───────────────────────────

def mutual_information(u_arr: np.ndarray, z_arr: np.ndarray,
                       n_u: int = 8, n_z: int = 128) -> float:
    """Empirical I(u ; z) in bits."""
    joint = np.zeros((n_u, n_z), dtype=float)
    for u, z in zip(u_arr, z_arr):
        joint[u, z] += 1.0
    joint /= joint.sum()

    p_u = joint.sum(axis=1, keepdims=True)
    p_z = joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    mi = float(np.sum(joint[mask] * np.log2(joint[mask] / (p_u * p_z)[mask])))
    return max(mi, 0.0)


def total_variation(u_arr: np.ndarray, uhat_arr: np.ndarray, z_arr: np.ndarray,
                    n_u: int = 8, n_z: int = 128) -> float:
    """Empirical total variation distance d_V(p_{u û z}, p*_{u û z})."""
    N = len(u_arr)
    joint3 = np.zeros((n_u, n_u, n_z), dtype=float)
    for u, uh, z in zip(u_arr, uhat_arr, z_arr):
        joint3[u, uh, z] += 1.0
    joint3 /= N

    p_u = joint3.sum(axis=(1, 2))
    p_z = joint3.sum(axis=(0, 1))

    ideal = np.zeros_like(joint3)
    for u in range(n_u):
        ideal[u, u, :] = p_u[u] * p_z

    return float(0.5 * np.sum(np.abs(joint3 - ideal)))


def security_bound(pe: float, mi: float) -> float:
    """Upper bound on ε-unconditional security."""
    return pe + np.sqrt(np.log(2) / 2.0 * mi)


def secrecy_capacity_awgn(snr_b_db: float, snr_e_db: float) -> float:
    """
    Theoretical secrecy capacity for the wiretap AWGN channel:
        C_s = max(0,  1/2 · log2((1 + SNR_B) / (1 + SNR_E)))
    """
    snr_b = 10 ** (snr_b_db / 10.0)
    snr_e = 10 ** (snr_e_db / 10.0)
    if snr_b <= snr_e:
        return 0.0
    return 0.5 * np.log2((1.0 + snr_b) / (1.0 + snr_e))


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(snr_b_db: float, snr_e_db: float, N: int = 2 ** 14,
             rng_seed: int = None):
    """
    Simulate N realisations of:
        RandomBinningEncoder  →  WiretapAWGN  →  RandomBinningDecoder

    Returns
    -------
    u_idx, uhat_idx, z_idx : np.ndarray of int, shape (N,)
        Integer representations (0-7) of message, decoded message, and
        Eve's received symbol index (0-127).
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    rng = np.random.default_rng(rng_seed)

    enc = RandomBinningEncoder(codebook=CODEBOOK, rng=rng)
    dec = RandomBinningDecoder(codebook=CODEBOOK)
    ch  = WiretapAWGN(snr_b_db, snr_e_db, lx=7, rng=rng)

    all_messages = [format(i, '03b') for i in range(8)]

    u_idx    = np.empty(N, dtype=int)
    uhat_idx = np.empty(N, dtype=int)
    z_idx    = np.empty(N, dtype=int)

    for i in range(N):
        # 1. Random message
        u_str = all_messages[rng.integers(8)]

        # 2. Encode to a 7-bit codeword
        x_str = enc.encode(u_str)
        x_int = np.array([int(x_str, 2)])  # shape (1,)

        # 3. Transmit through the AWGN channels
        y_int, z_int = ch.transmit(x_int)

        # 4. Decode Bob's received symbol
        y_str  = format(int(y_int[0]), '07b')
        uhat_str = dec.decode(y_str)

        u_idx[i]    = int(u_str, 2)
        uhat_idx[i] = int(uhat_str, 2)
        z_idx[i]    = int(z_int[0])

    return u_idx, uhat_idx, z_idx


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_task9(N: int = 2 ** 14):
    """
    Performs all sweeps and produces all plots required by Task 9.
    """

    # --- Fixed parameters for each sweep ---
    SNR_E_FIXED_DB = 5.0    # fixed Eve SNR while sweeping Bob's
    SNR_B_FIXED_DB = 15.0   # fixed Bob SNR while sweeping Eve's

    snrb_values = np.linspace(0, 25, 20)   # dB
    snre_values = np.linspace(0, 25, 20)   # dB

    # ── 1. Reliability sweep (vary SNR_B, fix SNR_E) ─────────────────────────
    print(f"\n  RELIABILITY SWEEP  (fixed SNR_E = {SNR_E_FIXED_DB} dB)")
    print(f"{'SNR_B':>7}  | {'P_e':>8}  | {'I(u;z)':>8}  | {'eps_sec':>8}  | {'d_V':>8}")
    print("-" * 55)

    pe_list, mi_snrb, dv_snrb, sec_snrb = [], [], [], []

    for snrb in snrb_values:
        u, uh, z = simulate(snrb, SNR_E_FIXED_DB, N)
        pe  = float(np.mean(u != uh))
        mi  = mutual_information(u, z)
        dv  = total_variation(u, uh, z)
        sec = security_bound(pe, mi)
        pe_list.append(pe); mi_snrb.append(mi)
        dv_snrb.append(dv); sec_snrb.append(sec)
        print(f"{snrb:>7.2f}  | {pe:>8.5f}  | {mi:>8.5f}  | {sec:>8.5f}  | {dv:>8.5f}")

    # ── 2. Secrecy sweep (vary SNR_E, fix SNR_B) ─────────────────────────────
    print(f"\n  SECRECY SWEEP  (fixed SNR_B = {SNR_B_FIXED_DB} dB)")
    print(f"{'SNR_E':>7}  | {'P_e':>8}  | {'I(u;z)':>8}  | {'eps_sec':>8}  | {'d_V':>8}  | {'C_s':>8}")
    print("-" * 67)

    mi_snre, dv_snre, sec_snre, cs_snre = [], [], [], []

    for snre in snre_values:
        u, uh, z = simulate(SNR_B_FIXED_DB, snre, N)
        pe  = float(np.mean(u != uh))
        mi  = mutual_information(u, z)
        dv  = total_variation(u, uh, z)
        sec = security_bound(pe, mi)
        cs  = secrecy_capacity_awgn(SNR_B_FIXED_DB, snre)
        mi_snre.append(mi);  dv_snre.append(dv)
        sec_snre.append(sec); cs_snre.append(cs)
        print(f"{snre:>7.2f}  | {pe:>8.5f}  | {mi:>8.5f}  | {sec:>8.5f}  | {dv:>8.5f}  | {cs:>8.5f}")

    # ── Theoretical secrecy capacity curves ──────────────────────────────────
    cs_vs_snrb = [secrecy_capacity_awgn(snrb, SNR_E_FIXED_DB) for snrb in snrb_values]
    cs_vs_snre = [secrecy_capacity_awgn(SNR_B_FIXED_DB, snre) for snre in snre_values]

    # ── Plotting ──────────────────────────────────────────────────────────────
    C_BOB, C_EVE = "steelblue", "tomato"
    C_SEC, C_DV, C_CS = "darkorange", "purple", "forestgreen"

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 2, hspace=0.50, wspace=0.35)

    # Plot 1: Reliability — P[û≠u] vs SNR_B
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(snrb_values, pe_list, 'o-', color=C_BOB, lw=2, ms=5,
                 label=r"$\hat{P}_e$ empirical")
    ax1.axvline(SNR_E_FIXED_DB, color='gray', ls='--', lw=1,
                label=rf"SNR$_E$ = {SNR_E_FIXED_DB} dB")
    ax1.set_xlabel(r"SNR$_B$ (dB)")
    ax1.set_ylabel(r"$P[\hat{u} \neq u]$")
    ax1.set_title(rf"Reliability vs SNR$_B$  (SNR$_E$ fixed = {SNR_E_FIXED_DB} dB)")
    ax1.legend(fontsize=8); ax1.grid(True, which='both', alpha=0.3)

    # Plot 2: Leaked info — I(u;z) vs SNR_E
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(snre_values, mi_snre, 'o-', color=C_EVE, lw=2, ms=5,
             label=r"$\hat{I}(u;z)$ empirical")
    ax2.axvline(SNR_B_FIXED_DB, color='gray', ls='--', lw=1,
                label=rf"SNR$_B$ = {SNR_B_FIXED_DB} dB")
    ax2.set_xlabel(r"SNR$_E$ (dB)")
    ax2.set_ylabel(r"$\hat{I}(u;\,z)$ [bits]")
    ax2.set_title(rf"Leaked information vs SNR$_E$  (SNR$_B$ fixed = {SNR_B_FIXED_DB} dB)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # Plot 3: Security bound vs SNR_B
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(snrb_values, sec_snrb, 's-', color=C_SEC, lw=2, ms=5,
             label=r"$\varepsilon_{sec}$ bound")
    ax3.plot(snrb_values, dv_snrb, '^-', color=C_DV, lw=2, ms=5,
             label=r"$d_V(p_{u\hat{u}z},\;p^*_{u\hat{u}z})$ empirical")
    ax3.axvline(SNR_E_FIXED_DB, color='gray', ls='--', lw=1)
    ax3.set_xlabel(r"SNR$_B$ (dB)")
    ax3.set_ylabel("Security metric")
    ax3.set_title(rf"Security bound vs SNR$_B$  (SNR$_E$ fixed = {SNR_E_FIXED_DB} dB)")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # Plot 4: Security bound vs SNR_E
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(snre_values, sec_snre, 's-', color=C_SEC, lw=2, ms=5,
             label=r"$\varepsilon_{sec}$ bound")
    ax4.plot(snre_values, dv_snre, '^-', color=C_DV, lw=2, ms=5,
             label=r"$d_V$ empirical")
    ax4.axvline(SNR_B_FIXED_DB, color='gray', ls='--', lw=1,
                label=rf"SNR$_B$ = {SNR_B_FIXED_DB} dB")
    ax4.set_xlabel(r"SNR$_E$ (dB)")
    ax4.set_ylabel("Security metric")
    ax4.set_title(rf"Security bound vs SNR$_E$  (SNR$_B$ fixed = {SNR_B_FIXED_DB} dB)")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    # Plot 5: Secrecy capacity vs SNR_B
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(snrb_values, cs_vs_snrb, '-', color=C_CS, lw=2,
             label=r"$C_s$ theory")
    ax5.axhline(3 / 7, color='navy', ls='--', lw=1.5,
                label=r"Scheme rate $= 3/7 \approx 0.43$ bit/use")
    ax5.axvline(SNR_E_FIXED_DB, color='gray', ls='--', lw=1)
    ax5.set_xlabel(r"SNR$_B$ (dB)")
    ax5.set_ylabel(r"$C_s$ [bit/channel use]")
    ax5.set_title(rf"Secrecy capacity vs SNR$_B$  (SNR$_E$ fixed = {SNR_E_FIXED_DB} dB)")
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

    # Plot 6: Secrecy capacity vs SNR_E
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(snre_values, cs_vs_snre, '-', color=C_CS, lw=2,
             label=r"$C_s$ theory")
    ax6.axhline(3 / 7, color='navy', ls='--', lw=1.5,
                label=r"Scheme rate $= 3/7 \approx 0.43$ bit/use")
    ax6.axvline(SNR_B_FIXED_DB, color='gray', ls='--', lw=1)
    ax6.set_xlabel(r"SNR$_E$ (dB)")
    ax6.set_ylabel(r"$C_s$ [bit/channel use]")
    ax6.set_title(rf"Secrecy capacity vs SNR$_E$  (SNR$_B$ fixed = {SNR_B_FIXED_DB} dB)")
    ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)

    fig.suptitle("Task 9 — System Security Evaluation over the Wiretap AWGN Channel",
                 fontsize=14, fontweight="bold")
    plt.savefig("Task9Image/task9_awgn_security.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved to Task9Image/task9_awgn_security.png")

    # ── Summary at a good working point ──────────────────────────────────────
    print(f"\nSUMMARY — SNR_B = {SNR_B_FIXED_DB} dB, SNR_E = {SNR_E_FIXED_DB} dB (secrecy-advantage regime)")
    u, uh, z = simulate(SNR_B_FIXED_DB, SNR_E_FIXED_DB, 2 ** 16)
    pe  = float(np.mean(u != uh))
    mi  = mutual_information(u, z)
    dv  = total_variation(u, uh, z)
    sec = security_bound(pe, mi)
    cs  = secrecy_capacity_awgn(SNR_B_FIXED_DB, SNR_E_FIXED_DB)
    print(f"  P[u_hat != u]     = {pe:.6f}")
    print(f"  I(u ; z)          = {mi:.6f} bits")
    print(f"  eps_sec (bound)   = {sec:.6f}")
    print(f"  d_V (empirical)   = {dv:.6f}")
    print(f"  C_s (theory)      = {cs:.6f} bit/channel use")
    print(f"  Scheme rate (3/7) = {3/7:.6f} bit/channel use")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs("Task9Image", exist_ok=True)
    np.random.seed(42)
    run_task9(N=2 ** 14)
