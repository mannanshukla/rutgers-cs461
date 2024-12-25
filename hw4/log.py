import numpy as np

# Data points
x = np.array([2, 1, -1, -2])

# Initial parameters
pi = [0.5, 0.5]   # Mixing coefficients
mu = [-1, 1]      # Means
sigma2 = [1, 1]   # Variances

# Gaussian PDF function
def gaussian_pdf(x, mu, sigma2):
    return (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-((x - mu) ** 2) / (2 * sigma2))

# Compute log-likelihood
log_likelihood = 0
for n in range(len(x)):  # Loop over data points
    likelihood_sum = 0
    for k in range(2):  # Loop over Gaussian components
        likelihood_sum += pi[k] * gaussian_pdf(x[n], mu[k], sigma2[k])
    log_likelihood += np.log(likelihood_sum)

print("Initial Log-Likelihood:", log_likelihood)
