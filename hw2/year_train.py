import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Dimensional Reduction and Whitening (5.1)
def perform_pca(logit, components=2):
    # Standardize the data (whitening)
    mean_logit = np.mean(logit, axis=0)
    std_logit = np.std(logit, axis=0)
    logit_scaled = (logit - mean_logit) / std_logit

    # PCA using eigen decomposition
    cov_matrix = np.cov(logit_scaled, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top `components` eigenvectors
    selected_vectors = eigenvectors[:, :components]

    # Transform data
    pca_transformed = logit_scaled @ selected_vectors
    
    return pca_transformed, mean_logit, std_logit, selected_vectors

# Function to create polynomial features manually
def create_polynomial_features(X, degree=2):
    """
    Generate polynomial features up to the given degree.
    X is a 2D array where each row is a sample and has two columns (x1, x2).
    Includes bias term.
    """
    poly_features = [np.ones(X.shape[0])]  # Bias term (constant 1)
    for d in range(1, degree + 1):
        for i in range(d + 1):
            poly_features.append((X[:, 0] ** (d - i)) * (X[:, 1] ** i))
    return np.vstack(poly_features).T

# Step 2: Train the Regression Model (5.2)
def train_model(train_logit, train_year, degree=2):
    # Apply PCA to reduce to 2-D
    pca_2d, mean_logit, std_logit, eigenvectors = perform_pca(train_logit, components=2)

    # Create polynomial features for the 2-D PCA-transformed data
    X_train = create_polynomial_features(pca_2d, degree=degree)
    y_train = train_year

    # Split the data into training and validation sets
    split_ratio = 0.8
    split_index = int(split_ratio * X_train.shape[0])
    X_train_data = X_train[:split_index]
    y_train_data = y_train[:split_index]
    X_val_data = X_train[split_index:]
    y_val_data = y_train[split_index:]

    # Train the regression model using Normal Equation: θ = (XᵀX)⁻¹Xᵀy
    theta = np.linalg.pinv(X_train_data.T @ X_train_data) @ X_train_data.T @ y_train_data

    # Predict on the validation set to evaluate performance
    y_val_pred = X_val_data @ theta
    val_mse = np.mean((y_val_data - y_val_pred) ** 2)
    print(f'Validation MSE for Degree {degree}: {val_mse}')

    return theta, mean_logit, std_logit, eigenvectors, val_mse

# Main Script Execution
if __name__ == "__main__":
    # Load the data from the correct path
    data = np.load("HW2_data/P5_data/vgg16_train.npz")
    train_logit = data["logit"]
    train_year = data["year"]

    # Test different polynomial degrees to observe overfitting/performance stagnation
    best_mse = float('inf')
    best_degree = None
    best_theta = None

    for degree in range(1, 6):
        print(f"Training model with polynomial degree {degree}...")
        theta, mean_logit, std_logit, eigenvectors, val_mse = train_model(train_logit, train_year, degree=degree)

        # Save model parameters if current model is the best
        if best_mse > val_mse:
            best_mse = val_mse
            best_degree = degree
            best_theta = theta
            np.save(f"HW2_data/P5_data/best_theta_degree_{degree}.npy", theta)
            np.savez("HW2_data/P5_data/pca_parameters.npz", mean=mean_logit, std=std_logit, eigenvectors=eigenvectors)
            print(f"Best model updated: Degree {degree}, MSE {best_mse}")

    print(f"Final Best Model: Degree {best_degree}, MSE {best_mse}")
