import numpy as np


def linear_kernel(X1, X2):
    """
    Linear kernel: K(x, y) = x^T · y
    
    This is equivalent to standard dot product and results in a linear decision boundary.
    
    Args:
        X1: First data matrix, shape (n_samples_1, n_features)
        X2: Second data matrix, shape (n_samples_2, n_features)
    
    Returns:
        Kernel matrix, shape (n_samples_1, n_samples_2)
    """
    return np.dot(X1, X2.T)


def polynomial_kernel(X1, X2, degree=3, coef0=1.0):
    """
    Polynomial kernel: K(x, y) = (x^T · y + c)^d
    
    Creates polynomial decision boundaries. Higher degree allows more complex boundaries
    but can lead to overfitting.
    
    Args:
        X1: First data matrix, shape (n_samples_1, n_features)
        X2: Second data matrix, shape (n_samples_2, n_features)
        degree: Polynomial degree (d), default=3
        coef0: Independent coefficient (c), default=1.0
    
    Returns:
        Kernel matrix, shape (n_samples_1, n_samples_2)
    """
    linear_term = np.dot(X1, X2.T)
    return (linear_term + coef0) ** degree


def rbf_kernel(X1, X2, gamma=0.1):
    """
    Radial Basis Function (RBF) / Gaussian kernel: K(x, y) = exp(-γ||x - y||²)
    
    Creates smooth, non-linear decision boundaries. The most popular kernel for SVM.
    
    Args:
        X1: First data matrix, shape (n_samples_1, n_features)
        X2: Second data matrix, shape (n_samples_2, n_features)
        gamma: Kernel coefficient (γ), default=0.1
               - Higher gamma: more complex (tight) boundaries, risk of overfitting
               - Lower gamma: smoother (loose) boundaries, risk of underfitting
    
    Returns:
        Kernel matrix, shape (n_samples_1, n_samples_2)
    """
    # Compute squared Euclidean distances efficiently
    # ||x - y||² = ||x||² + ||y||² - 2x^T·y
    
    X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)  # Shape: (n_samples_1, 1)
    X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)  # Shape: (1, n_samples_2)
    
    # Compute pairwise squared distances
    squared_distances = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
    
    # Apply RBF transformation
    return np.exp(-gamma * squared_distances)


def sigmoid_kernel(X1, X2, gamma=0.1, coef0=0.0):
    """
    Sigmoid kernel: K(x, y) = tanh(γ·x^T·y + c)
    
    Similar to neural network activation. Can behave like a two-layer perceptron.
    
    Args:
        X1: First data matrix, shape (n_samples_1, n_features)
        X2: Second data matrix, shape (n_samples_2, n_features)
        gamma: Kernel coefficient (γ), default=0.1
        coef0: Independent coefficient (c), default=0.0
    
    Returns:
        Kernel matrix, shape (n_samples_1, n_samples_2)
    """
    linear_term = np.dot(X1, X2.T)
    return np.tanh(gamma * linear_term + coef0)


def create_kernel_function(kernel_type, **params):
    """
    Factory function to create kernel functions with specific parameters.
    
    This is useful for creating kernel functions with fixed parameters
    that can be passed to kernelized models.
    
    Args:
        kernel_type: String specifying kernel type
                     Options: 'linear', 'polynomial', 'rbf', 'sigmoid'
        **params: Kernel-specific parameters
    
    Returns:
        Kernel function that takes (X1, X2) and returns kernel matrix
    """
    if kernel_type == 'linear':
        return lambda X1, X2: linear_kernel(X1, X2)
    
    elif kernel_type == 'polynomial':
        degree = params.get('degree', 3)
        coef0 = params.get('coef0', 1.0)
        return lambda X1, X2: polynomial_kernel(X1, X2, degree=degree, coef0=coef0)
    
    elif kernel_type == 'rbf':
        gamma = params.get('gamma', 0.1)
        return lambda X1, X2: rbf_kernel(X1, X2, gamma=gamma)
    
    elif kernel_type == 'sigmoid':
        gamma = params.get('gamma', 0.1)
        coef0 = params.get('coef0', 0.0)
        return lambda X1, X2: sigmoid_kernel(X1, X2, gamma=gamma, coef0=coef0)
    
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}. "
                        f"Choose from: 'linear', 'polynomial', 'rbf', 'sigmoid'")


def compute_kernel_matrix(X, kernel_func):
    """
    Compute kernel matrix K where K[i,j] = kernel(X[i], X[j]).
    
    This is a convenience function for computing the full kernel matrix
    for a single dataset (useful for visualization and analysis).
    
    Args:
        X: Data matrix, shape (n_samples, n_features)
        kernel_func: Kernel function
    
    Returns:
        Kernel matrix, shape (n_samples, n_samples)
    """
    return kernel_func(X, X)