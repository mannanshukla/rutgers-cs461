import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(X, y, w):
    z = np.dot(X, w)
    predictions = sigmoid(z)
    nll = -np.sum(y * np.log(predictions + 1e-10) + (1 - y) * np.log(1 - predictions + 1e-10))
    return nll

def compute_gradient(X, y, w):
    z = np.dot(X, w)
    predictions = sigmoid(z)
    gradient = np.dot(X.T, (predictions - y))
    return gradient

def train_logistic_regression(X, y, learning_rate=1e-1, tolerance=1e-10, max_iter=100000):
    w = np.zeros(X.shape[1])  # Initialize weights to zeros
    prev_nll = float('inf')  # Initialize previous NLL as infinity

    for i in range(max_iter):
        nll = loss(X, y, w)  # Compute current NLL
        
        if abs(prev_nll - nll) < tolerance:
            print(f"Converged after {i} iterations with NLL = {nll:.6f}")
            break
        
        gradient = compute_gradient(X, y, w)  # Compute gradient
        w -= learning_rate * gradient  # Update weights
        prev_nll = nll  # Update previous NLL
        
        if i % 1000 == 0:
            print(f"Iteration {i}: NLL = {nll:.6f}")
    
    return w

def main():
    train_file = './HW3_data/P4_files/train4_2.npz'
    test_file = './HW3_data/P4_files/test4_2.npz'
    weights_file = './HW3_data/P4_files/q4weights.npy'

    # Load training data
    train = np.load(train_file)
    X_train = train['x']
    y_train = train['y']

    # Load testing data
    test = np.load(test_file)
    X_test = test['x']
    y_test = test['y']

    # Train logistic regression
    print("Training logistic regression...")
    w = train_logistic_regression(X_train, y_train)

    # Evaluate accuracy on training data
    train_predictions = np.round(sigmoid(np.dot(X_train, w)))
    train_accuracy = np.mean(train_predictions == y_train)
    print(f"Training Accuracy: {train_accuracy:.4f}")

    # Evaluate accuracy on testing data
    test_predictions = np.round(sigmoid(np.dot(X_test, w)))
    test_accuracy = np.mean(test_predictions == y_test)
    print(f"Testing Accuracy: {test_accuracy:.4f}")

    # Save weights
    np.save(weights_file, w)
    print(f"Weights saved to {weights_file}")

if __name__ == "__main__":
    main()

