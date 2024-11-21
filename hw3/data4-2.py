import pandas as pd
import numpy as np


def compute_pca(data, num_components=50):
    """
    Perform Principal Component Analysis (PCA) on the given data.
    Args:
        data (numpy.ndarray): The input data matrix (n_samples, n_features).
        num_components (int): Number of principal components to retain.
    Returns:
        reduced_data (numpy.ndarray): Data transformed to the reduced dimensionality.
        mean_vector (numpy.ndarray): The mean vector of the original data.
    """
    # Center the data
    mean_vector = data.mean(axis=0)
    centered_data = data - mean_vector

    # Singular Value Decomposition (SVD)
    u, s, vt = np.linalg.svd(centered_data, full_matrices=False)

    # Reduce to the desired number of components
    u_reduced = u[:, :num_components]
    s_reduced = np.diag(s[:num_components])

    # Project the data onto the top principal components
    reduced_data = np.dot(u_reduced, s_reduced)
    return reduced_data, mean_vector


def save_splits(data, labels, train_size=3500, filepaths=None):
    """
    Shuffle and split the data into training and testing sets, and save to files.
    Args:
        data (numpy.ndarray): The reduced data matrix (n_samples, n_features).
        labels (numpy.ndarray): The corresponding labels.
        train_size (int): Number of samples to allocate to the training set.
        filepaths (dict): Dictionary of filepaths for saving splits.
    """
    # Shuffle the data and labels
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)

    data_shuffled = data[indices]
    labels_shuffled = labels[indices]

    # Split into training and testing sets
    X_train, X_test = data_shuffled[:train_size], data_shuffled[train_size:]
    y_train, y_test = labels_shuffled[:train_size], labels_shuffled[train_size:]

    # Save the splits to files
    np.savez(filepaths["train"], x=X_train, y=y_train)
    np.savez(filepaths["test"], x=X_test, y=y_test)

    print(f"Data splits saved: {filepaths['train']}, {filepaths['test']}")


def main():
    # Load the dataset
    data_path = './HW3_data/P4_files/spam_ham.csv'
    df = pd.read_csv(data_path)

    # Drop unnecessary columns
    df.drop(columns=['Unnamed: 0'], inplace=True)

    # Extract features and labels
    features = df.drop(columns=['cls']).values
    labels = df['cls'].values

    # Perform PCA
    reduced_data, mean_vec = compute_pca(features, num_components=50)

    # Save the splits
    filepaths = {
        "train": "./HW3_data/P4_files/train4_2.npz",
        "test": "./HW3_data/P4_files/test4_2.npz"
    }
    save_splits(reduced_data, labels, train_size=3500, filepaths=filepaths)


if __name__ == "__main__":
    main()
