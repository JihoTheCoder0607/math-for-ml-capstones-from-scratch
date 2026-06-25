import numpy as np

# helper functions
def mean(X):
    mean = np.sum(X, axis=0) / X.shape[0]
    return mean

def variance(X):
    X_mean = mean(X)
    X_centered = X-X_mean
    variance = np.sum(X_centered ** 2, axis=0) / (X.shape[0]-1)
    return variance

def std(X):
    X_var = variance(X)
    std = np.sqrt(X_var)
    return std

def covariance(X, Y):
    X_mean = mean(X)
    X_centered = X-X_mean
    Y_mean = mean(Y)
    Y_centered = Y - Y_mean
    cov = np.sum(X_centered * Y_centered)/(X.shape[0]-1)
    return cov

# PCA class
class pca:
    def __init__(self, n_components):
        self.n_components = n_components
    def standardize(self, X):
        X_std = std(X)
        X_mean = mean(X)
        return (X - X_mean)/X_std
    def covariance_matrix(self, X):
        cov_matrix = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[1]):
            x = X[:, i]
            for j in range(X.shape[1]):
                y = X[:, j]
                cov = covariance(x, y)
                cov_matrix[i][j] = cov
        return cov_matrix





