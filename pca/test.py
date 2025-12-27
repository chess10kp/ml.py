from sklearn.datasets import load_wine
import torch
import numpy as np

# get the data
X, y = load_wine(return_X_y=True)
X_torch = torch.from_numpy(X)
y_torch = torch.from_numpy(y)


# scale the data
mean = torch.mean(X_torch, axis=0)
std = torch.std(X_torch, axis=0)

scaled_X: torch.Tensor = (X_torch - mean) / std

# covariance matrix
# measure of how close two features are

cov = (scaled_X.T @ scaled_X) / (scaled_X.shape[0] -1)

values, vectors = torch.linalg.eigh(cov)

idx = torch.argsort(values, descending=True)
values = values[idx]

vectors = vectors[:, idx]

# get first two

pca_components = vectors[: , 2]

pca_x = scaled_X @ pca_components