import numpy as np

# Data points
x = np.array([2, 1, -1, -2])

# Given parameters
pi = [0.5, 0.5]  # Mixing coefficients
mu = [-1, 1]     # Means
sigma2 = [1, 1]  # Variances

# Gaussian PDF function
def gaussian_pdf(x, mu, sigma2):
    return (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-((x - mu) ** 2) / (2 * sigma2))

# E-step: compute responsibilities (gamma)
gamma = np.zeros((len(x), 2))  # Responsibilities for 2 Gaussians
for k in range(2):  # For each Gaussian
    gamma[:, k] = pi[k] * gaussian_pdf(x, mu[k], sigma2[k])
gamma /= gamma.sum(axis=1, keepdims=True)  # Normalize

print("Responsibilities (gamma):")
print(gamma)

# M-step: Update parameters
N = len(x)

# Update pi (mixing coefficients)
pi_new = gamma.sum(axis=0) / N

# Update mu (means)
mu_new = np.sum(gamma * x[:, np.newaxis], axis=0) / gamma.sum(axis=0)

# Update sigma^2 (variances)
sigma2_new = np.sum(gamma * (x[:, np.newaxis] - mu_new) ** 2, axis=0) / gamma.sum(axis=0)

print("Updated parameters:")
print("Pi (mixing coefficients):", pi_new)
print("Mu (means):", mu_new)
print("Sigma^2 (variances):", sigma2_new)

# Compute log-likelihood
log_likelihood = np.sum(np.log(np.sum([
    pi_new[k] * gaussian_pdf(x, mu_new[k], sigma2_new[k]) for k in range(2)
], axis=0)))

print("Log-Likelihood:", log_likelihood)
