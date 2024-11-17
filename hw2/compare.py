import numpy as np
import matplotlib.pyplot as plt

degree = 9

# Function to prepare polynomial features
def prep_data(X, degree):
    X = np.vstack([X**i for i in range(degree + 1)]).T
    return X

# Load OLS models (w(λ = 0)) and Ridge models (best λ)
ols_models = []
for i in range(5):
    model = np.load(f"weights_mmse_fold_{i}.npy")
    ols_models.append(model)

# Load Ridge models from the best λ identified
ridge_models = []
best_lambda = 0.0001  # Replace with your best λ
for i in range(5):
    model = np.load(f"ridge_weights_lambda_{best_lambda}_fold_{i}.npy")
    ridge_models.append(model)

# Average the models
ols_avg = np.mean(ols_models, axis=0)
ridge_avg = np.mean(ridge_models, axis=0)

# Print shapes to verify compatibility
print(f"OLS Average Weights Shape: {ols_avg.shape}")
print(f"Ridge Average Weights Shape: {ridge_avg.shape}")

# Prepare plot data
X = np.linspace(0, 1, 100)
X_poly = prep_data(X, degree)

# Plot the two models over the range 0 ≤ x ≤ 1
plt.figure()
plt.plot(X, np.dot(X_poly, ols_avg), label="OLS", color='blue')
plt.plot(X, np.dot(X_poly, ridge_avg), label="Ridge", color='red')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Comparison of OLS and Ridge Regression Models")
plt.legend()
plt.grid(True)
plt.show()

# Evaluate models on the test dataset
test_data = np.load("HW2_data/P3_data/test.npz")
test_X = test_data["x"]
test_y = test_data["y"]

# Prepare test data
test_X_poly = prep_data(test_X, degree)

# Calculate test errors for both models
ols_test_error = np.mean((test_y - np.dot(test_X_poly, ols_avg)) ** 2)
ridge_test_error = np.mean((test_y - np.dot(test_X_poly, ridge_avg)) ** 2)

print(f"OLS Test Error: {ols_test_error}")
print(f"Ridge Test Error: {ridge_test_error}")
