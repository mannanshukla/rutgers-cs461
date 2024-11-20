import numpy as np

# Load the test data
data_test = np.load('./HW3_data/P3_data/data_2/test.npz')
x_test, y_test = data_test['x'], data_test['y']

# Precomputed statistics (from train3-4.py)
mu_pos = np.array([0.0130754, 0.06295251])  # Mean for Class +1
cov_pos = np.array([[0.98285498, 0.00612046], [0.00612046, 1.05782804]])  # Covariance for Class +1
mu_neg = np.array([-0.02313942, -0.02114952])  # Mean for Class -1
cov_neg = np.array([[1.00329037, -0.01142356], [-0.01142356, 4.97693356]])  # Covariance for Class -1

# Priors (optional; compute from training data if needed)
prior_pos = 0.5  # Example prior for Class +1 (replace if known)
prior_neg = 0.5  # Example prior for Class -1 (replace if known)

# Multivariate Gaussian PDF function
def multivariate_gaussian_pdf(x, mu, cov):
    d = len(mu)
    cov_inv = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    factor = 1 / np.sqrt((2 * np.pi) ** d * det_cov)
    diff = x - mu
    exp_term = np.exp(-0.5 * diff.T @ cov_inv @ diff)
    return factor * exp_term

# Predict labels
predictions = []
for x in x_test:
    # Compute posterior probabilities for both classes
    p_pos = multivariate_gaussian_pdf(x, mu_pos, cov_pos) * prior_pos
    p_neg = multivariate_gaussian_pdf(x, mu_neg, cov_neg) * prior_neg

    # Assign the label based on higher posterior probability
    predictions.append(1 if p_pos >= p_neg else -1)

# Convert predictions to a NumPy array
predictions = np.array(predictions)

# Calculate accuracy
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy (2D GDA): {accuracy:.4f}")

