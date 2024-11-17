import numpy as np

train_data = np.load("HW2_data/P3_data/train.npz")
train_x, train_y = train_data["x"], train_data["y"]

def create_polynomial_features(x, degree):
    return np.vstack([x**i for i in range(degree + 1)]).T

def ols_regression(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cross_validation_ols(train_x, train_y, degree, k_folds=5):
    fold_size = len(train_x) // k_folds
    indices = np.arange(len(train_x))
    mse_scores = []
    models_ordinary_mmse = []

    for fold in range(k_folds):
        valid_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.delete(indices, valid_idx)

        X_train = create_polynomial_features(train_x[train_idx], degree)
        y_train = train_y[train_idx]
        X_valid = create_polynomial_features(train_x[valid_idx], degree)
        y_valid = train_y[valid_idx]

        weights = ols_regression(X_train, y_train)
        
        models_ordinary_mmse.append(weights)
        np.save(f"weights_mmse_fold_{fold}.npy", weights)
        y_pred = X_valid @ weights
        mse_scores.append(mean_squared_error(y_valid, y_pred))

    return np.mean(mse_scores), models_ordinary_mmse

degree = 9

train_X = create_polynomial_features(train_x, degree)

avg_validation_mse, models_ordinary_mmse = cross_validation_ols(train_x, train_y, degree)

print(f"Average Validation MSE (5-fold): {avg_validation_mse}")

final_weights = ols_regression(train_X, train_y)

np.save("final_weights_ols.npy", final_weights)
