import numpy as np

# Your data matrix
Phi = np.array([
    [1, 4, 1, 1],
    [1, 7, 0, 2],
    [1, 10, 1, 3],
    [1, 13, 0, 4]
])

# Calculate Phi^T * Phi
Phi_T_Phi = np.dot(Phi.T, Phi)

# Check if determinant is non-zero
det = np.linalg.det(Phi_T_Phi)
print(f"Determinant: {det}")

# If determinant is zero, it's not invertible
if det != 0:
    print("Matrix is invertible.")
else:
    print("Matrix is not invertible.")
