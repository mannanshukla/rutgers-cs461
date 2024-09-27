import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

def calculate_pmf(M):
    # Calculate probabilities for each possible outcome
    x = np.arange(M + 1)
    pmf = binom.pmf(x, M, 0.5)
    return x, pmf

def plot_pmf(M_values):
    plt.figure(figsize=(15, 5))
    
    for i, M in enumerate(M_values, 1):
        x, pmf = calculate_pmf(M)
        
        plt.subplot(1, 3, i)
        plt.bar(x, pmf, alpha=0.8)
        plt.title(f'PMF for M = {M}')
        plt.xlabel('Final Position')
        plt.ylabel('Probability')
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Calculate and plot PMFs for M = 5, 10, 100
M_values = [5, 10, 100]
plot_pmf(M_values)

# Print some statistics to observe the change
for M in M_values:
    x, pmf = calculate_pmf(M)
    mean = np.sum(x * pmf)
    variance = np.sum((x - mean)**2 * pmf)
    std_dev = np.sqrt(variance)
    print(f"M = {M}:")
    print(f"  Mean: {mean:.2f}")
    print(f"  Standard Deviation: {std_dev:.2f}")
    print(f"  Variance: {variance:.2f}")
    print()
