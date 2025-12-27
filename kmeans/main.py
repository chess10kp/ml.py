#!/usr/bin/env python
# coding: utf-8

# In[13]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import random


# In[14]:


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
)

train_dataset = MNIST(root="data", train=True, transform=transform, download=True)


# In[ ]:


n_clusters = 10
iters = 1000
tol = 10e-4
clusters = None
X = None

def initialize_clusters(clusters, n_clusters, train):
    if clusters is None:
        train_dataset = train
        idx = torch.randperm(len(train_dataset))[:n_clusters]
        clusters = torch.stack([train_dataset[i][0] for i in idx])

    return clusters


# In[16]:


centroids = initialize_clusters(clusters, n_clusters, train_dataset)
assert(centroids.shape[0] == n_clusters)


# In[17]:


X = torch.stack([x[0] for x in train_dataset])
Y = torch.tensor([x[1] for x in train_dataset] )
assert(X.shape[0] == Y.shape[0])
assert(len(X.shape) == 2)
assert(len(Y.shape) == 1)
X.shape, Y.shape, centroids.shape


# In[18]:


def distance(X : torch.stack, centroids: torch.stack) -> torch.Tensor:
    return torch.sum((X[:, None, :] - centroids[None, :, :]) ** 2, dim=2)

def group_into_centroids(X, centroids):
    initial_distances = distance(X, centroids)
    return torch.argmin(initial_distances, 1)

labels = group_into_centroids(X, centroids)


# In[19]:


labels.shape


# In[20]:


# recompute centroids as mean of points in each cluster 
# filter for each cluster, the points that are in that cluster 
i = 0 
while i < iters:
    i += 1
    labels = group_into_centroids(X, centroids)
    updated_centroids = []
    for j in range(0, n_clusters):
        clustered = X[labels == j]
        if len(clustered) == 0:
            updated_centroids.append(centroids[j])
        else:
            m = clustered.mean(dim=0)
            updated_centroids.append(m) 
    updated_centroids = torch.stack(updated_centroids) 
    if torch.allclose(updated_centroids, centroids, atol=tol):
        break 

    centroids = updated_centroids
centroids


# In[21]:


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

final_labels = group_into_centroids(X, centroids)

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(centroids[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'Cluster {i}')
    ax.axis('off')
plt.show()


# In[22]:


from sklearn.metrics import confusion_matrix
import numpy as np

conf = confusion_matrix(Y, final_labels)
purity = np.sum(np.max(conf, axis=0)) / np.sum(conf)
print(f"Cluster purity: {purity:.4f}")

