import pandas as pd
import numpy as np


def compute_pca(data, num_components=50):
    mean_vector = data.mean(axis=0)
    centered_data = data - mean_vector

    u, s, vt = np.linalg.svd(centered_data, full_matrices=False)

    u_reduced = u[:, :num_components]
    s_reduced = np.diag(s[:num_components])

    reduced_data = np.dot(u_reduced, s_reduced)
    return reduced_data, mean_vector


def save_splits(data, labels, train_size=3500, filepaths=None):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)

    data_shuffled = data[indices]
    labels_shuffled = labels[indices]

    X_train, X_test = data_shuffled[:train_size], data_shuffled[train_size:]
    y_train, y_test = labels_shuffled[:train_size], labels_shuffled[train_size:]

    np.savez(filepaths["train"], x=X_train, y=y_train)
    np.savez(filepaths["test"], x=X_test, y=y_test)

    print(f"Data splits saved: {filepaths['train']}, {filepaths['test']}")


def main():
    data_path = './HW3_data/P4_files/spam_ham.csv'
    df = pd.read_csv(data_path)

    df.drop(columns=['Unnamed: 0'], inplace=True)

    features = df.drop(columns=['cls']).values
    labels = df['cls'].values

    reduced_data, mean_vec = compute_pca(features, num_components=50)

    filepaths = {
        "train": "./HW3_data/P4_files/train4_2.npz",
        "test": "./HW3_data/P4_files/test4_2.npz"
    }
    save_splits(reduced_data, labels, train_size=3500, filepaths=filepaths)


if __name__ == "__main__":
    main()
