import numpy as np
import matplotlib.pyplot as plt

train_data = np.load("HW2_data/P3_data/train.npz")
train_x, train_y = train_data["x"], train_data["y"]

def create_polynomial_features(x, degree):
    return np.vstack([x**i for i in range(degree + 1)]).T

def ridge_regression(X, y, lambd):
    identity = np.eye(X.shape[1])
    return np.linalg.inv(X.T @ X + lambd * identity) @ X.T @ y

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cross_validation_ridge(train_x, train_y, degree, lambdas, k_folds=5):
    fold_size = len(train_x) // k_folds
    indices = np.arange(len(train_x))
    avg_mse_per_lambda = []
    best_models = []
    ridge_models = []

    for lambd in lambdas:
        mse_scores = []
        fold_models = []
        for fold in range(k_folds):
            valid_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            train_idx = np.delete(indices, valid_idx)

            X_train = create_polynomial_features(train_x[train_idx], degree)
            y_train = train_y[train_idx]
            X_valid = create_polynomial_features(train_x[valid_idx], degree)
            y_valid = train_y[valid_idx]

            weights = ridge_regression(X_train, y_train, lambd)

            fold_models.append(weights)
            if lambd == 0:
                np.save(f"weights_mmse_fold_{fold}.npy", weights)
            else:
                np.save(f"ridge_weights_lambda_{lambd}_fold_{fold}.npy", weights)

            y_pred = X_valid @ weights
            mse_scores.append(mean_squared_error(y_valid, y_pred))

        avg_mse_per_lambda.append(np.mean(mse_scores))
        ridge_models.append(fold_models)

        if lambd == lambdas[np.argmin(avg_mse_per_lambda)]:
            best_models = fold_models

    return avg_mse_per_lambda, best_models

degree = 9
lambdas = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.5, 1.0]

avg_mse_per_lambda, best_ridge_models = cross_validation_ridge(train_x, train_y, degree, lambdas)

best_lambda = lambdas[np.argmin(avg_mse_per_lambda)]

train_X = create_polynomial_features(train_x, degree)
best_weights = ridge_regression(train_X, train_y, best_lambda)
np.save("best_weights_ridge.npy", best_weights)

plt.plot(lambdas, avg_mse_per_lambda, marker='o', linestyle='-', label="Validation Error")
plt.xlabel("Lambda (λ)")
plt.ylabel("Average Validation Error (MSE)")
plt.title("Validation Error vs Lambda")
plt.xscale('log')
plt.grid(True)

for i, (lambd, mse) in enumerate(zip(lambdas, avg_mse_per_lambda)):
    plt.text(lambd, mse, f"{lambd:.1e}", fontsize=8, ha='right', va='bottom')

plt.legend()
plt.show()

print(f"Best lambda (λ*): {best_lambda}")
