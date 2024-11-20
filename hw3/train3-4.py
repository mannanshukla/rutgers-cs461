import numpy as np

# Load the training data
data = np.load('./HW3_data/P3_data/data_2/train.npz')
x, y = data['x'], data['y']  # x is 2D, y is the label (+1 or -1)

# Separate data by class
x_pos = x[y == 1]  # Class +1
x_neg = x[y == -1]  # Class -1

# Compute mean vector (mu) for each class
mu_pos = np.mean(x_pos, axis=0)
mu_neg = np.mean(x_neg, axis=0)

# Compute covariance matrix (Sigma) for each class
cov_pos = np.cov(x_pos, rowvar=False)  # Covariance matrix for Class +1
cov_neg = np.cov(x_neg, rowvar=False)  # Covariance matrix for Class -1

# Print the results
print(f"Class +1: Mean = {mu_pos}, Covariance =\n{cov_pos}")
print(f"Class -1: Mean = {mu_neg}, Covariance =\n{cov_neg}")

