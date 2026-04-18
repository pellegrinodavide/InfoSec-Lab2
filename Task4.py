import numpy as np
import matplotlib.pyplot as plt
from Task1 import WiretapUniformErrorChannel
from Task2 import RandomBinningEncoder
from Task3 import RandomBinningDecoder

def run_reliability_test(iterations):
    enc = RandomBinningEncoder()
    dec = RandomBinningDecoder(enc.codebook)
    
    # We test Bob's channel with varying max errors 'r'
    # Although the handout uses r=1, we can test r=0, 1, 2, 3 to see the breakdown
    r_values = [0, 1, 2, 3]
    error_probabilities = []

    for r in r_values:
        ch = WiretapUniformErrorChannel(n=7, r=r, s=3)
        errors = 0
        
        for _ in range(iterations):
            # 1. Generate random message u
            u_int = np.random.randint(0, 8)
            u_str = format(u_int, '03b')
            
            # 2. Encode
            x_str = enc.encode(u_str)
            x_int = int(x_str, 2)
            
            # 3. Transmit through Bob's channel
            y_arr, _ = ch.transmit(x_int, N=1)
            y_str = format(y_arr[0], '07b')
            
            # 4. Decode
            u_hat = dec.decode(y_str)
            
            # 5. Check for error
            if u_hat != u_str:
                errors += 1
        
        prob = errors / iterations
        error_probabilities.append(prob)
        print(f"Max Errors r={r} | Decoding Error Probability: {prob:.4f}")

    return r_values, error_probabilities

# ── Execution and Plotting ──────────────────────────────────────────────────
if __name__ == "__main__":
    r_vals, p_errors = run_reliability_test(2**14)

    plt.figure(figsize=(8, 5))
    plt.plot(r_vals, p_errors, marker='o', linestyle='-', color='blue')
    plt.title("System Reliability: Decoding Error Probability vs. Channel Noise")
    plt.xlabel("Maximum Channel Errors (r)")
    plt.ylabel(r"$P(\hat{u} \neq u)$")
    plt.grid(True, alpha=0.3)
    plt.xticks(r_vals)
    plt.show()