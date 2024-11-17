import numpy as np
import matplotlib.pyplot as plt

degree = 9

def prep_data(X, degree):
    X = np.vstack([X**i for i in range(degree + 1)]).T
    return X

def ols_regression(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

train_data = np.load("HW2_data/P3_data/train_100.npz")
train_x = train_data["x"]
train_y = train_data["y"]

train_X_poly = prep_data(train_x, degree)

weights_ols = ols_regression(train_X_poly, train_y)

np.save("weights_ols_large.npy", weights_ols)

X_plot = np.linspace(0, 1, 100)
X_plot_poly = prep_data(X_plot, degree)

y_plot_ols = np.dot(X_plot_poly, weights_ols)

plt.figure()
plt.scatter(train_x, train_y, color='orange', label="Data Points", alpha=0.5)
plt.plot(X_plot, y_plot_ols, label="OLS - Large Dataset", color='blue')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("OLS Regression Model on Large Dataset with Data Points")
plt.legend()
plt.grid(True)
plt.show()

print("Trained OLS model weights saved to weights_ols_large.npy")
