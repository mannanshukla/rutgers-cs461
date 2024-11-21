import numpy as np

try:
    data = np.load('./HW3_data/P3_data/data_1/train.npz') 
    x, y = data['x'], data['y']
except FileNotFoundError:
    print("File './data_1/train.npz' not found. Ensure the file path is correct.")
    exit()
except AttributeError as e:
    print(f"AttributeError: {e}")
    exit()

x_pos = x[y == 1]
x_neg = x[y == -1]

mu_pos = np.mean(x_pos)
var_pos = np.var(x_pos)

mu_neg = np.mean(x_neg)
var_neg = np.var(x_neg)

print(f"Class +1: Mean = {mu_pos:.4f}, Variance = {var_pos:.4f}")
print(f"Class -1: Mean = {mu_neg:.4f}, Variance = {var_neg:.4f}")

