import numpy as np

# Load test data
filepath = './HW3_data/P3_data/data_2/test.npz'  # Consistent file path
test_data = np.load(filepath)
X = test_data['x']  # Test features
y = test_data['y']  # True labels

# Function to compute the Gaussian likelihood
def gaussian_pdf(x, mean, cov):
    """
    Computes the likelihood of x under a multivariate Gaussian distribution.
    Args:
        x: Data points (n_samples, n_features)
        mean: Mean vector (n_features,)
        cov: Covariance matrix (n_features, n_features)
    Returns:
        Likelihood values (n_samples,)
    """
    diff = x - mean
    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)
    
    # Avoid numerical instability by adding a small value to the determinant
    cov_det = max(cov_det, 1e-6)
    
    exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
    normalization = np.sqrt((2 * np.pi) ** x.shape[1] * cov_det)
    return np.exp(exponent) / normalization

# Define means and covariances
mean_pos = np.array([0, 0])         # Mean for Class +1
mean_neg_1 = np.array([0, 2])       # Mean for Class -1 (component 1)
mean_neg_2 = np.array([0, -2])      # Mean for Class -1 (component 2)
cov_pos = np.eye(2)                 # Covariance for Class +1
cov_neg = np.eye(2)                 # Covariance for Class -1

# Compute likelihoods
likelihood_pos = gaussian_pdf(X, mean_pos, cov_pos)
likelihood_neg_1 = gaussian_pdf(X, mean_neg_1, cov_neg)
likelihood_neg_2 = gaussian_pdf(X, mean_neg_2, cov_neg)

# Mixture model for Class -1
likelihood_neg = 0.5 * likelihood_neg_1 + 0.5 * likelihood_neg_2

# Predict class based on posterior probabilities
# Assuming equal priors, posterior is proportional to the likelihood
pred = np.where(likelihood_pos >= likelihood_neg, 1, -1)

# Compute and print the accuracy
accuracy = np.mean(pred == y)
print(f"Test Accuracy: {accuracy:.4f}")

