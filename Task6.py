# Task 6: Implement the wiretap binary symmetric channel

import numpy as np
import matplotlib.pyplot as plt


# --- Start Wiretap BSC ---
class WiretapBSC:
    """
    A Wiretap Binary Symmetric Channel (BSC) that operates on 7-bit words.
    It simulates communication from a sender to Bob (the legitimate receiver) and Eve (the eavesdropper).
    - Bob's channel is a BSC with a crossover probability of epsilon.
    - Eve's channel is a BSC with a crossover probability of delta.
    - The two channels are conditionally independent given the transmitted word x.
    """
    # --- Start __init__ ---
    def __init__(self, epsilon: float, delta: float, n: int = 7):
        """
        Initializes the WiretapBSC.
        Args:
            epsilon: The crossover probability for Bob's channel.
            delta: The crossover probability for Eve's channel.
            n: The word length in bits (default is 7).
        """
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")
        if not (0.0 <= delta <= 1.0):
            raise ValueError("delta must be in [0, 1]")
        self.epsilon = epsilon
        self.delta = delta
        self.n = n
    # --- End __init__ ---

    # --- Start Single BSC Pass ---
    def _bsc_str(self, x_str: str, p: float) -> str:
        """
        Simulates passing a single bit-string through a BSC with crossover probability p.
        Each bit is flipped independently with probability p. This is achieved by
        generating a Bernoulli(p) flip mask and XORing it with the input.
        Args:
            x_str: The input bit-string.
            p: The crossover probability.
        Returns: The received word as a bit-string.
        """
        flip = (np.random.rand(self.n) < p).astype(int)
        x_bits = np.array(list(x_str), dtype=int)
        y_bits = x_bits ^ flip  # XOR: bit is flipped if flip==1
        return ''.join(map(str, y_bits))
    # --- End Single BSC Pass ---

    # --- Start Batch Transmit ---
    def transmit_str(self, x_str: str):
        """
        Transmits a single word (as a bit-string) through both Bob's and Eve's channels.
        Args:
            x_str: The word to transmit.
        Returns: A tuple (y_str, z_str) containing the words received by Bob and Eve.
        """
        y = self._bsc_str(x_str, self.epsilon)
        z = self._bsc_str(x_str, self.delta)
        return y, z
    # --- End Batch Transmit ---

    # --- Start Vectorized Batch Transmit (for Task 7) ---
    def transmit_batch(self, x: np.ndarray):
        """
        Transmits a batch of words through both channels simultaneously.
        This method is vectorized for efficiency.
        Args:
            x: A NumPy array of shape (N, n) representing N words of n bits each.
        Returns:
            A tuple (y, z) where:
            - y is a NumPy array of shape (N, n) for words received by Bob.
            - z is a NumPy array of shape (N, n) for words received by Eve.
        """
        flip_y = (np.random.rand(*x.shape) < self.epsilon).astype(int)
        flip_z = (np.random.rand(*x.shape) < self.delta).astype(int)
        return x ^ flip_y, x ^ flip_z
    # --- End Vectorized Batch Transmit (for Task 7) ---
# --- End Wiretap BSC ---

# --- Start BER Verification ---
def verify_channel(n_words: int = 2**16):
    """
    Verifies the channel implementation by comparing theoretical and measured Bit Error Rates (BER).
    For several (epsilon, delta) pairs, this function transmits a large number
    of random words and calculates the BER for both Bob's and Eve's channels.
    Args: n_words: The number of random words to transmit for the verification.
    """
    test_pairs = [
        (0.05, 0.20),
        (0.10, 0.30),
        (0.15, 0.40),
        (0.25, 0.45),
        (0.01, 0.49),
    ]

    print("\nTask 6 — Wiretap BSC verification\n")
    print(f"{'(eps, delta)':<13} | {'BER Bob':>10}    |  {'BER Eve':>4}")
    print("-" * 42)

    eps_theory, eps_measured = [], []
    dlt_theory, dlt_measured = [], []

    for eps, dlt in test_pairs:
        ch = WiretapBSC(epsilon=eps, delta=dlt)

        # Generate random n-bit words as a NumPy array for efficiency.
        x = np.random.randint(0, 2, size=(n_words, ch.n))
        y, z = ch.transmit_batch(x)

        ber_b = float(np.mean(x != y))
        ber_e = float(np.mean(x != z))

        eps_theory.append(eps)
        eps_measured.append(ber_b)
        dlt_theory.append(dlt)
        dlt_measured.append(ber_e)

        print(f"({eps:.2f}, {dlt:.2f})  |    {ber_b:.6f}   |  {ber_e:.6f}")

    # --- Plotting Results ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, theory, measured, label, color in zip(
        axes,
        [eps_theory, dlt_theory],
        [eps_measured, dlt_measured],
        ["Bob (epsilon)", "Eve (delta)"],
        ["steelblue", "tomato"],
    ):
        ax.plot([0, 0.5], [0, 0.5], "k--", lw=1, label="Ideal (measured = theory)")
        ax.scatter(theory, measured, color=color, zorder=5, s=60, label="Measured BER")
        ax.set_xlabel("Theoretical crossover probability")
        ax.set_ylabel("Measured BER")
        ax.set_title(f"BSC verification – {label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Task 6: Wiretap BSC – BER verification", fontweight="bold")
    plt.tight_layout()
    plt.savefig("Task6Image/Task6_verification.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nPlot saved to Task6_verification.png")
# --- End BER Verification ---

# --- Start Main Execution ---
if __name__ == "__main__":
    np.random.seed(42)
    verify_channel()
# --- End Main Execution ---