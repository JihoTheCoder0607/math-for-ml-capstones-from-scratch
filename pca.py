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

def dot(u, v):
    assert u.shape == v.shape
    result = 0
    for i in range(u.shape[0]):
        result += u[i] * v[i]
    return result

def norm(v):
    squared_sum = 0
    for i in range(v.shape[0]):
        squared_sum += v[i] ** 2
    return squared_sum ** 0.5

def normalize(v):
    norm_v = norm(v)
    if norm_v < 1e-12:
        raise ValueError("Zero Vector Detected")
    return v / norm_v

def projection(vector, onto):
    return dot(vector, onto) / dot(onto, onto) * onto

def gram_schmidt(A):
    Q = [normalize(A[:, 0])]
    for i in range(1, A.shape[1]):
        a = A[:, i]
        u = a.astype(float).copy()
        for q in Q:
            u -= projection(u, q)
        q = normalize(u)
        Q.append(q)
    Q = np.column_stack(Q)
    R = Q.T @ A
    return Q, R

# Verifying for Gram-Schmidt
A = np.array([[1, 2], [3, 4]])
Q, R = gram_schmidt(A)
print(np.allclose(Q @ R, A))

# Verifying Q orthogonality
print(np.allclose(Q.T @ Q, np.eye(Q.shape[1])))

# Verifying that R is triangular
print(R)

# QR Algorithm using Classical Gram-Schmidt
def qr_algorithm(A, max_iter=1000, tol=1e-12):
    A = A.astype(float).copy()
    Q_total = np.eye(A.shape[0])
    for _ in range(max_iter):
        Q, R = gram_schmidt(A)
        A = R @ Q
        Q_total = Q_total @ Q
        if np.all(np.abs(np.tril(A, k=-1)) < tol):
            break
    eigenvalues = np.diag(A)
    eigenvectors = Q_total
    return eigenvalues, eigenvectors

# PCA class
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
    def standardize(self, X):
        return (X - self.mean_)/self.std_
    def covariance_matrix(self, X):
        cov_matrix = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[1]):
            x = X[:, i]
            for j in range(X.shape[1]):
                y = X[:, j]
                cov = covariance(x, y)
                cov_matrix[i][j] = cov
        return cov_matrix
    def eigen(self, A, max_iter=1000, tol=1e-12):
        return qr_algorithm(A, max_iter, tol)
    def select_components(self, eigenvalues, eigenvectors):
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        return eigenvalues[:self.n_components], eigenvectors[:, :self.n_components]
    def project(self, X, W):
        return X @ W
    def fit_transform(self, X, max_iter=1000, tol=1e-12):
        self.mean_ = mean(X)
        self.std_ = std(X)

        X = self.standardize(X)
        cov_matrix = self.covariance_matrix(X)
        eigenvalues, eigenvectors = self.eigen(cov_matrix, max_iter, tol)
        top_eigenvalues, top_eigenvectors = self.select_components(eigenvalues, eigenvectors)
        self.eigenvalues_ = top_eigenvalues
        self.components_ = top_eigenvectors
        X_projected = self.project(X, top_eigenvectors)
        return X_projected
    def reconstruct(self, X_projected):
        return (X_projected @ self.components_.T) * self.std_ + self.mean_
