import numpy as np

# Load training and testing data
train_data = np.load('./HW3_data/P4_files/train4_2.npz')
test_data = np.load('./HW3_data/P4_files/test4_2.npz')

X_train = train_data['x']  # Training features
y_train = train_data['y']  # Training labels
X_test = test_data['x']    # Testing features
y_test = test_data['y']    # Testing labels

# Load the trained weights
w = np.load('./HW3_data/P4_files/q4weights.npy')

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Prediction function
def predict(X, w):
    """
    Predict class labels using the logistic regression model.
    Args:
        X: Input features (n_samples, n_features).
        w: Trained weights (n_features,).
    Returns:
        Predictions (0 or 1 for each sample).
    """
    probabilities = sigmoid(X @ w)
    return (probabilities >= 0.5).astype(int)

# Compute predictions
y_train_pred = predict(X_train, w)
y_test_pred = predict(X_test, w)

# Calculate accuracies
train_accuracy = np.mean(y_train_pred == y_train)
test_accuracy = np.mean(y_test_pred == y_test)

# Print the results
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
