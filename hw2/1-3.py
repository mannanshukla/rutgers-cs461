import numpy as np

# Your data matrix
Phi = np.array([
    [1, 4, 1, 1],
    [1, 7, 0, 2],
    [1, 10, 1, 3],
    [1, 13, 0, 4]
])

# The target values vector
y = np.array([16, 23, 36, 43])

# Calculate Phi^T * Phi
Phi_T_Phi = np.dot(Phi.T, Phi)

# Calculate the inverse of (Phi^T * Phi)
Phi_T_Phi_inv = np.linalg.inv(Phi_T_Phi)

# Calculate Phi^T * y
Phi_T_y = np.dot(Phi.T, y)

# Calculate the weights w
w = np.dot(Phi_T_Phi_inv, Phi_T_y)

# Print the coefficients
print("Estimated coefficients:", w)
