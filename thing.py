import numpy as np

# Given data
E_X = np.array([0.2, 0.3, 0.1])
cov_X_X = np.array([[2.75, 0.43, 0],
                    [0.43, 2.25, 0],
                    [0, 0, 1]])

# Step 1: Compute the eigenvalue decomposition of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_X_X)

# Step 2: Calculate the inverse square root of the eigenvalues
inv_sqrt_eigenvalues = np.diag(1.0 / np.sqrt(eigenvalues))

# Step 3: Compute the whitening matrix A
A = eigenvectors @ inv_sqrt_eigenvalues @ eigenvectors.T

# Step 4: Calculate the mean-shifting vector b
b = -A @ E_X

# Output results
print("Whitening matrix A:\n", A)
print("Mean-shifting vector b:\n", b)

# Optional: Applying the whitening transformation on new data X
X = np.random.multivariate_normal(E_X, cov_X_X, 10000)  # Simulating 10,000 3D data points
Y = A @ X.T + b[:, np.newaxis]  # Applying whitening

# Verify that Y has zero mean and identity covariance
print("Mean of Y:", np.mean(Y, axis=1))
print("Covariance of Y:\n", np.cov(Y))

