import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.load("HW2_data/P5_data/vgg16_train.npz")
train_logit = data["logit"]
train_year = data["year"]

mean_logit = np.mean(train_logit, axis=0)
std_logit = np.std(train_logit, axis=0)
logit_scaled = (train_logit - mean_logit) / std_logit

cov_matrix = np.cov(logit_scaled, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, sorted_indices]

pca_1d = logit_scaled @ eigenvectors[:, :1]
pca_2d = logit_scaled @ eigenvectors[:, :2]

plt.figure(figsize=(8, 6))
plt.scatter(pca_1d, train_year, c=train_year, cmap='viridis', s=2)
plt.colorbar(label='Year')
plt.xlabel('PCA Component 1')
plt.ylabel('Year')
plt.title('Year over M=1 PCA')
plt.show()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(pca_2d[:, 0], pca_2d[:, 1], train_year, c=train_year, cmap='viridis', s=2)
fig.colorbar(sc, label='Year')

ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('Year')
ax.set_title('Year over M=2 PCA (3D View)')
plt.show()
