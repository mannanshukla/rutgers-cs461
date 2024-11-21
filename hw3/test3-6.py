import numpy as np

filepath = './HW3_data/P3_data/data_2/test.npz'
test_data = np.load(filepath)
X = test_data['x']  
y = test_data['y']

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
    
    cov_det = max(cov_det, 1e-6)
    
    exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
    normalization = np.sqrt((2 * np.pi) ** x.shape[1] * cov_det)
    return np.exp(exponent) / normalization

mean_pos = np.array([0, 0])         # Mean for Class +1
mean_neg_1 = np.array([0, 2])       # Mean for Class -1 (component 1)
mean_neg_2 = np.array([0, -2])      # Mean for Class -1 (component 2)
cov_pos = np.eye(2)                 # Covariance for Class +1
cov_neg = np.eye(2)                 # Covariance for Class -1

likelihood_pos = gaussian_pdf(X, mean_pos, cov_pos)
likelihood_neg_1 = gaussian_pdf(X, mean_neg_1, cov_neg)
likelihood_neg_2 = gaussian_pdf(X, mean_neg_2, cov_neg)

likelihood_neg = 0.5 * likelihood_neg_1 + 0.5 * likelihood_neg_2

pred = np.where(likelihood_pos >= likelihood_neg, 1, -1)

accuracy = np.mean(pred == y)
print(f"Test Accuracy: {accuracy:.4f}")

