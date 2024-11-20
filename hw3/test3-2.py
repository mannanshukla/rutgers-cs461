import numpy as np

# Load the test data
data = np.load('./HW3_data/P3_data/data_1/test.npz')
x_test, y_test = data['x'], data['y']

# Precomputed statistics
mu_pos = -0.0722  # Mean for Class +1
var_pos = 1.3031  # Variance for Class +1
mu_neg = 0.9402   # Mean for Class -1
var_neg = 1.9426  # Variance for Class -1

# Gaussian PDF function
def gaussian_pdf(x, mu, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mu) ** 2) / (2 * var))

# Predict labels
predictions = []
for x in x_test:
    # Compute likelihoods for both classes
    p_pos = gaussian_pdf(x, mu_pos, var_pos)
    p_neg = gaussian_pdf(x, mu_neg, var_neg)
    
    # Assign the label based on higher likelihood
    predictions.append(1 if p_pos >= p_neg else -1)

# Convert predictions to a NumPy array
predictions = np.array(predictions)

# Calculate accuracy
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy:.4f}")

