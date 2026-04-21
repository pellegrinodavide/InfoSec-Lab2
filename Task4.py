import numpy as np
import matplotlib.pyplot as plt
from Task1 import WiretapUniformErrorChannel
from Task2 import RandomBinningEncoder
from Task3 import RandomBinningDecoder


def get_pz_u_matrix(encoder, channel, n_trials=2**14):
    # Message d in the rows, z values in the columns (0 to 127)
    pz_u_matrix = np.zeros((8, 128))

    for d_idx in range(8):
        d_str = format(d_idx, '03b')
        z_samples = []
        
        for _ in range(n_trials):
            # The encoder internally picks codeword 1 or 2 randomly
            x_str = encoder.encode(d_str)
            _, z_val = channel.transmit(int(x_str, 2), N=1)
            z_samples.append(z_val)
        
        # Calculate empirical PMF for message d
        counts, _ = np.histogram(z_samples, bins=np.arange(129))
        pmf = counts / n_trials
        pz_u_matrix[d_idx, :] = pmf # Save for Mutual Info calculation
    return pz_u_matrix


def plot_pmf(pmf_matrix):
    fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
    fig.suptitle(r"Empirical Conditional PMF $p_{z|u}(\cdot|d)$ for all $d \in \mathcal{M}$", 
                 fontsize=16, fontweight='bold', y=0.95)
    
    axes = axes.flatten()
    x_axis = np.arange(128)

    for d in range(8):
        ax = axes[d]
        d_str = format(d, '03b')
        
        # Use a bar plot with specific width and alignment
        ax.bar(x_axis, pmf_matrix[d], color='navy', width=0.8, alpha=0.7, label='Empirical Data')
        
        ax.set_title(f"Message $d = {d_str}$", fontsize=12, loc='left')
        ax.set_ylim(0, 0.015) 
        ax.set_ylabel("Probability")
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        
        if d >= 6: # Only label the bottom x-axes
            ax.set_xlabel("Eavesdropper Received Value $z$ (decimal)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()
    

def get_emp_joint_pmf(pz_u_matrix):
    p_d = 1/8 # Assuming uniform distribution over messages d
    joint_pmf = pz_u_matrix * p_d
    return joint_pmf
    

def get_emp_mutual_info(joint_pmf_uz, p_u = None, p_z = None):
    if p_u is None:
        p_u = np.sum(joint_pmf_uz, axis=1) # Marginal over z
    if p_z is None:
        p_z = np.sum(joint_pmf_uz, axis=0) # Marginal over u
    
    I_u_z = 0.0
    for d in range(8):
        for z in range(128):
            p_dz = joint_pmf_uz[d, z]
            if p_dz > 0 and p_u[d] > 0 and p_z[z] > 0:
                I_u_z += p_dz * np.log2(p_dz / (p_u[d] * p_z[z]))
    
    return I_u_z
    
if __name__ == "__main__":    
    N_TRIALS = 2**14 
    enc = RandomBinningEncoder()
    ch = WiretapUniformErrorChannel(n=7, r=1, s=3)

    # Get the empirical conditional PMF matrix p_{z|u}(z|d) for all messages d
    pz_u_matrix = get_pz_u_matrix(enc, ch, N_TRIALS)
    # Run the plot
    plot_pmf(pz_u_matrix)
    
    print("\n--------------------------------------------------------------\n")
    
    p_cd_matrix = get_emp_joint_pmf(pz_u_matrix)
    
    
    # Marginal distribution of the messages d (p_u)
    p_u_marginal = np.sum(p_cd_matrix, axis=1)
    print("Marginal p_u:")
    for d in range(8):
        print(f"  d = {format(d, '03b')}  →  p_u(d) = {p_u_marginal[d]:.6f}") 
        
        
    # Marginal distribution of the eavesdropper's received values z (p_z)
    p_z_marginal = np.sum(p_cd_matrix, axis=0)
    print("Marginal p_z:")
    for z in range(128):
        print(f"  z = {z:03d}  →  p_z(z) = {p_z_marginal[z]:.6f}")
        
    # Calculate the entropy H(u)
    H_u = -np.sum(p_u_marginal * np.log2(p_u_marginal))
    print(f"\nEntropy H(u) = {H_u:.6f} bits")
    
    # Calculate the empirical mutual information I(u;z)
    I_u_z = get_emp_mutual_info(p_cd_matrix, p_u_marginal, p_z_marginal)
    print(f"\nEmpirical Mutual Information I(u;z) = {I_u_z:.6f} bits")
