import numpy as np

data = np.load('./HW3_data/P3_data/data_1/test.npz')
x_test, y_test = data['x'], data['y']

mu_pos = -0.0722  # Mean for Class +1
var_pos = 1.3031  # Variance for Class +1
mu_neg = 0.9402   # Mean for Class -1
var_neg = 1.9426  # Variance for Class -1

def gaussian_pdf(x, mu, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mu) ** 2) / (2 * var))

predictions = []
for x in x_test:
    p_pos = gaussian_pdf(x, mu_pos, var_pos)
    p_neg = gaussian_pdf(x, mu_neg, var_neg)
    
    predictions.append(1 if p_pos >= p_neg else -1)

predictions = np.array(predictions)

accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy:.4f}")

