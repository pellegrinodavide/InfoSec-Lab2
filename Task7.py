# Task 7: Evaluate the system security over the Wiretap BSC

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from Task2 import RandomBinningEncoder, CODEBOOK
from Task3 import RandomBinningDecoder
from Task6 import WiretapBSC


#  Information-Theoretic Helpers
def h2(p: float) -> float:
    """
    Computes the binary entropy function h2(p).
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)

def secrecy_capacity_bsc(epsilon: float, delta: float) -> float:
    """
    Calculates the theoretical secrecy capacity for the wiretap BSC.
    """
    return max(0.0, h2(delta) - h2(epsilon))

def mutual_information(u_arr: np.ndarray, z_arr: np.ndarray,
                       n_u: int = 8, n_z: int = 128) -> float:
    """
    Computes the empirical mutual information I_hat(u ; z) in bits.
    """
    # Build the empirical joint distribution
    joint = np.zeros((n_u, n_z), dtype=float)
    for u, z in zip(u_arr, z_arr):
        joint[u, z] += 1.0
    joint /= joint.sum()

    p_u = joint.sum(axis=1, keepdims=True)   # Marginal distribution of u
    p_z = joint.sum(axis=0, keepdims=True)   # Marginal distribution of z

    # Only sum over non-zero entries to avoid log(0) errors
    mask = joint > 0
    mi = float(np.sum(joint[mask] * np.log2(joint[mask] / (p_u * p_z)[mask])))
    return max(mi, 0.0)


def total_variation(u_arr: np.ndarray, uhat_arr: np.ndarray, z_arr: np.ndarray,
                    n_u: int = 8, n_z: int = 128) -> float:
    """
    Computes the empirical total variation distance (d_V).
    """
    N = len(u_arr)

    # Compute the empirical 3-way joint distribution
    joint3 = np.zeros((n_u, n_u, n_z), dtype=float)
    for u, uh, z in zip(u_arr, uhat_arr, z_arr):
        joint3[u, uh, z] += 1.0
    joint3 /= N

    p_u = joint3.sum(axis=(1, 2))   # Marginal distribution over u
    p_z = joint3.sum(axis=(0, 1))   # Marginal distribution over z

    # Build the ideal joint distribution
    ideal = np.zeros_like(joint3)
    for u in range(n_u):
        ideal[u, u, :] = p_u[u] * p_z   # Only the diagonal where u == u_hat has mass

    return float(0.5 * np.sum(np.abs(joint3 - ideal)))


def security_bound(pe: float, mi: float) -> float:
    """
    Computes the upper bound on epsilon-unconditional security.
    """
    return pe + np.sqrt(np.log(2) / 2.0 * mi)


#  Simulation Function
def simulate(epsilon: float, delta: float, N: int = 2**14,
             rng_seed: int = None):
    """
    Simulates N realizations of the full communication chain:
        RandomBinningEncoder -> WiretapBSC -> RandomBinningDecoder
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    # Initialize components
    enc = RandomBinningEncoder(codebook=CODEBOOK)
    dec = RandomBinningDecoder(codebook=CODEBOOK)
    ch  = WiretapBSC(epsilon=epsilon, delta=delta)

    u_idx    = np.empty(N, dtype=int)
    uhat_idx = np.empty(N, dtype=int)
    z_idx    = np.empty(N, dtype=int)

    all_messages = [format(i, '03b') for i in range(8)]  # List of strings '000' to '111'

    for i in range(N):
        # 1. Draw a random message
        u_str = all_messages[np.random.randint(8)]

        # 2. Encode to a 7-bit codeword
        x_str = enc.encode(u_str)

        # 3. Transmit through the BSC channels
        y_str, z_str = ch.transmit_str(x_str)

        # 4. Decode Bob's received word
        uhat_str = dec.decode(y_str)

        # Convert strings to integers and store the results
        u_idx[i]    = int(u_str, 2)
        uhat_idx[i] = int(uhat_str, 2)
        z_idx[i]    = int(z_str, 2)

    return u_idx, uhat_idx, z_idx

#  Main Execution - Sweeps and Plots
def run_task7(N: int = 2**14):
    """
    Executes the main task logic.
    """
    # --- Sweep Parameters ---
    DELTA_FIXED   = 0.45
    epsilons      = np.linspace(0.01, 0.49, 20)

    EPSILON_FIXED = 0.10
    deltas        = np.linspace(0.01, 0.49, 20)

    # --- 1. Reliability Sweep ---
    print(f"\n       RELIABILITY SWEEP (fixed delta = {DELTA_FIXED})")
    print(f"{'eps':>5}  |  {'P_e':>5}    | {'I(u;z)':>7}  | {'eps_sec':>8}  |   {'d_V':>3}")
    print("-" * 54)

    pe_list, mi_eps, dv_eps, sec_eps = [], [], [], []

    for eps in epsilons:
        u, uh, z = simulate(eps, DELTA_FIXED, N)
        pe  = float(np.mean(u != uh))
        mi  = mutual_information(u, z)
        dv  = total_variation(u, uh, z)
        sec = security_bound(pe, mi)
        pe_list.append(pe); mi_eps.append(mi)
        dv_eps.append(dv);  sec_eps.append(sec)
        print(f"{eps:6.3f} | {pe:8.5f}  | {mi:8.5f} | {sec:8.5f}  | {dv:8.5f}")

    # --- 2. Secrecy Sweep ---
    print(f"\n\n              SECRECY SWEEP (fixed epsilon = {EPSILON_FIXED})")
    print(f"{'dlt':>5}  |  {'P_e':>5}    | {'I(u;z)':>7}  | {'eps_sec':>8}  |    {'d_V':>3}    |   {'C_s':>3}")
    print("-" * 66)

    mi_dlt, dv_dlt, sec_dlt, cs_list = [], [], [], []

    for dlt in deltas:
        u, uh, z = simulate(EPSILON_FIXED, dlt, N)
        pe  = float(np.mean(u != uh))
        mi  = mutual_information(u, z)
        dv  = total_variation(u, uh, z)
        sec = security_bound(pe, mi)
        cs  = secrecy_capacity_bsc(EPSILON_FIXED, dlt)
        mi_dlt.append(mi);  dv_dlt.append(dv)
        sec_dlt.append(sec); cs_list.append(cs)
        print(f"{dlt:6.3f} | {pe:8.5f}  | {mi:8.5f} | {sec:8.5f}  | {dv:8.5f}  | {cs:8.5f}")

    # --- Theoretical Curves ---
    cs_vs_eps = [secrecy_capacity_bsc(e, DELTA_FIXED) for e in epsilons]
    cs_vs_dlt = [secrecy_capacity_bsc(EPSILON_FIXED, d) for d in deltas]

    # --- Plotting ---
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 2, hspace=0.48, wspace=0.35)

    C_BOB, C_EVE, C_SEC, C_DV, C_CS = "steelblue", "tomato", "darkorange", "purple", "green"

    # Plot 1: Reliability vs epsilon
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epsilons, pe_list, "o-", color=C_BOB, lw=2, ms=5, label=r"$\hat{P}_e$ empirical")
    ax1.axvline(DELTA_FIXED, color="gray", ls="--", lw=1, label=rf"$\delta = {DELTA_FIXED}$")
    ax1.set_xlabel(r"$\varepsilon$ (Bob's crossover probability)")
    ax1.set_ylabel(r"$P[\hat{u} \neq u]$")
    ax1.set_title(rf"Reliability vs $\varepsilon$ ($\delta$ fixed = {DELTA_FIXED})")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # Plot 2: Leaked mutual information vs delta
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(deltas, mi_dlt, "o-", color=C_EVE, lw=2, ms=5, label=r"$\hat{I}(u;z)$ empirical")
    ax2.axvline(EPSILON_FIXED, color="gray", ls="--", lw=1, label=rf"$\varepsilon = {EPSILON_FIXED}$")
    ax2.set_xlabel(r"$\delta$ (Eve's crossover probability)")
    ax2.set_ylabel(r"$\hat{I}(u;\,z)$ [bits]")
    ax2.set_title(rf"Leaked information vs $\delta$ ($\varepsilon$ fixed = {EPSILON_FIXED})")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # Plot 3: Security bound vs epsilon
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epsilons, sec_eps, "s-", color=C_SEC, lw=2, ms=5, label=r"$\varepsilon_{sec}$ bound")
    ax3.plot(epsilons, dv_eps, "^-", color=C_DV, lw=2, ms=5, label=r"$d_V(p_{u\hat{u}z},\;p^*_{u\hat{u}z})$ empirical")
    ax3.set_xlabel(r"$\varepsilon$")
    ax3.set_ylabel("Security metric")
    ax3.set_title(rf"Security bound vs $\varepsilon$ ($\delta$ fixed = {DELTA_FIXED})")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # Plot 4: Security bound vs delta
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(deltas, sec_dlt, "s-", color=C_SEC, lw=2, ms=5, label=r"$\varepsilon_{sec}$ bound")
    ax4.plot(deltas, dv_dlt, "^-", color=C_DV, lw=2, ms=5, label=r"$d_V$ empirical")
    ax4.axvline(EPSILON_FIXED, color="gray", ls="--", lw=1, label=rf"$\varepsilon = {EPSILON_FIXED}$")
    ax4.set_xlabel(r"$\delta$")
    ax4.set_ylabel("Security metric")
    ax4.set_title(rf"Security bound vs $\delta$ ($\varepsilon$ fixed = {EPSILON_FIXED})")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    # Plot 5: Secrecy capacity vs epsilon
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(epsilons, cs_vs_eps, "-", color=C_CS, lw=2, label=r"$C_s = [h_2(\delta)-h_2(\varepsilon)]^+$ theory")
    ax5.set_xlabel(r"$\varepsilon$")
    ax5.set_ylabel(r"$C_s$ [bit/channel use]")
    ax5.set_title(rf"Secrecy capacity vs $\varepsilon$ ($\delta$ fixed = {DELTA_FIXED})")
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

    # Plot 6: Secrecy capacity vs delta
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(deltas, cs_vs_dlt, "-", color=C_CS, lw=2, label=r"$C_s$ theory")
    ax6.axhline(3 / 7, color="navy", ls="--", lw=1.5, label=r"Scheme rate $= 3/7 \approx 0.43$ bit/use")
    ax6.axvline(EPSILON_FIXED, color="gray", ls="--", lw=1)
    ax6.set_xlabel(r"$\delta$")
    ax6.set_ylabel(r"$C_s$ [bit/channel use]")
    ax6.set_title(rf"Secrecy capacity vs $\delta$ ($\varepsilon$ fixed = {EPSILON_FIXED})")
    ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)

    fig.suptitle("Task 7: System Security Evaluation over the Wiretap BSC", fontsize=14, fontweight="bold")
    plt.savefig("Task7Image/Task7_BSC_security.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nPlot saved to Task7_BSC_security.png")

    # --- Summary at Working Point (epsilon=0.10, delta=0.45) ---
    print("SUMMARY — epsilon=0.10, delta=0.45 (secrecy-advantage regime)")
    u, uh, z = simulate(0.10, 0.45, 2**16)
    pe  = float(np.mean(u != uh))
    mi  = mutual_information(u, z)
    dv  = total_variation(u, uh, z)
    sec = security_bound(pe, mi)
    cs  = secrecy_capacity_bsc(0.10, 0.45)
    
    print(f"  P[u_hat != u]     = {pe:.6f}")
    print(f"  I(u ; z)          = {mi:.6f} bits")
    print(f"  eps_sec (bound)   = {sec:.6f}")
    print(f"  d_V (empirical)   = {dv:.6f}")
    print(f"  C_s (theory)      = {cs:.6f} bit/channel use")
    print(f"  Scheme rate (3/7) = {3/7:.6f} bit/channel use")


# --- Main Execution ---
if __name__ == "__main__":
    np.random.seed(42)
    run_task7(N=2**14)