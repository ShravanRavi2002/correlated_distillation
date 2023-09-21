import numpy as np
import math
import torch
import random
import pandas as pd
from torch.utils.data import Dataset

def generate_color_vectors(C: int, d: int, d_prime: int):

  s = 0.0

  beta = np.zeros((C, d))
  a = np.zeros(C)

  B = np.random.rand(d, d_prime)
  B = np.divide(B, np.linalg.norm(B, 'fro')) * np.sqrt(d*d_prime)

  for c in range(C):
    mean = np.array([0] * d_prime)
    cov = np.divide(np.identity(d_prime), d_prime)
    v_c = np.random.multivariate_normal(mean, cov)
    beta_c = np.matmul(B, v_c)
    for i in range(d):
      beta[c][i] = beta_c[i]

    a[c] = -1.75 * np.linalg.norm(beta_c) / math.sqrt(d)
    s += np.linalg.norm(beta_c, ord=2) ** 2

  return beta, a

class CorrelatedDataset(Dataset):
    def __init__(self, C=500, d=500, d_prime=10, num_samples=100):
        self.C = C
        self.d = d
        self.d_prime = d_prime
        self.num_samples = num_samples

        self.b, self.a = generate_color_vectors(C, d, d_prime)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        mean = np.array([0] * self.d)
        cov = np.divide(np.identity(self.d), self.d)
        index = []

        x = np.random.multivariate_normal(mean, cov)
        for c in range(self.b.shape[0]):
          if (x.dot(self.b[c]) + self.a[c] > 0):
            index.append(c)

        label = np.array([0] * self.b.shape[0])
        for i in index:
          label[i] = 1
        return torch.from_numpy(x).to(torch.float64), torch.from_numpy(label).to(torch.float64)


class CorrelatedMissingDataset(Dataset):
    def __init__(self, C=500, d=500, d_prime=10, num_samples=100):
        self.C = C
        self.d = d
        self.d_prime = d_prime
        self.num_samples = num_samples

        self.b, self.a = generate_color_vectors(C, d, d_prime)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        mean = np.array([0] * self.d)
        cov = np.divide(np.identity(self.d), self.d)
        index = []
        threshold = 0.5

        x = np.random.multivariate_normal(mean, cov)
        for c in range(self.b.shape[0]):
          if (x.dot(self.b[c]) + self.a[c] > 0):
            # take random subset of labels
            if np.random.uniform() < threshold:
              index.append(c)

        label = np.array([0] * self.b.shape[0])
        for i in index:
          label[i] = 1
        return torch.from_numpy(x).to(torch.float64), torch.from_numpy(label).to(torch.float64)