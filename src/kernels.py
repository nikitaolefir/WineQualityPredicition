import numpy as np


def linear_kernel(X1, X2):
    """
    Linear kernel - just the standard dot product between samples.
    Formula: K(x, y) = x^T · y
    
    This is equivalent to not using a kernel at all - it works in the
    original feature space. Results in linear decision boundaries.
    
    Args:
        X1: first data matrix, shape (n_samples_1, n_features)
        X2: second data matrix, shape (n_samples_2, n_features)
    
    Returns:
        Kernel matrix, shape (n_samples_1, n_samples_2)
    """
    # Computing dot product between all pairs of samples
    return np.dot(X1, X2.T)


def polynomial_kernel(X1, X2, degree=3, coef0=1.0):
    """
    Polynomial kernel - creates polynomial decision boundaries.
    Formula: K(x, y) = (x^T · y + c)^d
    
    The degree parameter controls the complexity of the boundary:
    - degree=2: quadratic boundaries (curves, ellipses)
    - degree=3: cubic boundaries (more complex curves)
    - higher degree: even more complex but risk of overfitting
    
    Args:
        X1: first data matrix, shape (n_samples_1, n_features)
        X2: second data matrix, shape (n_samples_2, n_features)
        degree: polynomial degree (d), default=3
        coef0: independent coefficient (c), default=1.0
    
    Returns:
        Kernel matrix, shape (n_samples_1, n_samples_2)
    """
    # First computing the linear kernel (dot product)
    linear_term = np.dot(X1, X2.T)
    
    # Adding the coefficient and raising to power
    return (linear_term + coef0) ** degree


def rbf_kernel(X1, X2, gamma=0.1):
    """
    Radial Basis Function (RBF) kernel, also called Gaussian kernel.
    Formula: K(x, y) = exp(-γ||x - y||²)
    
    This is the most popular kernel for SVM. It creates smooth, non-linear
    decision boundaries. The gamma parameter is crucial:
    - Higher gamma (e.g., 1.0): tight, complex boundaries around each point
      → Can lead to overfitting
    - Lower gamma (e.g., 0.01): smooth, simple boundaries
      → May underfit if too low
    
    Args:
        X1: first data matrix, shape (n_samples_1, n_features)
        X2: second data matrix, shape (n_samples_2, n_features)
        gamma: kernel coefficient (γ), default=0.1
    
    Returns:
        Kernel matrix, shape (n_samples_1, n_samples_2)
    """
    # Computing squared Euclidean distances efficiently
    # Using the identity: ||x - y||² = ||x||² + ||y||² - 2x^T·y
    # This is faster than computing distances directly
    
    # Computing squared norms for X1 samples
    X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)  # Shape: (n_samples_1, 1)
    
    # Computing squared norms for X2 samples
    X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)  # Shape: (1, n_samples_2)
    
    # Computing pairwise squared distances
    # Broadcasting makes this work for all pairs at once
    squared_distances = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
    
    # Applying the exponential transformation
    # exp(-gamma * distance²) gives values between 0 and 1
    # Values close to 1 mean the samples are very similar
    return np.exp(-gamma * squared_distances)


def sigmoid_kernel(X1, X2, gamma=0.1, coef0=0.0):
    """
    Sigmoid kernel - behaves similarly to a neural network activation.
    Formula: K(x, y) = tanh(γ·x^T·y + c)
    
    This kernel can sometimes behave like a two-layer neural network.
    However, it's not positive semi-definite for all parameter values,
    which can cause issues. Less commonly used than RBF or polynomial.
    
    Args:
        X1: first data matrix, shape (n_samples_1, n_features)
        X2: second data matrix, shape (n_samples_2, n_features)
        gamma: kernel coefficient (γ), default=0.1
        coef0: independent coefficient (c), default=0.0
    
    Returns:
        Kernel matrix, shape (n_samples_1, n_samples_2)
    """
    # Computing linear term first
    linear_term = np.dot(X1, X2.T)
    
    # Applying sigmoid transformation (tanh)
    # Output is between -1 and 1
    return np.tanh(gamma * linear_term + coef0)


def create_kernel_function(kernel_type, **params):
    """
    Factory function for creating kernel functions with fixed parameters.
    
    This is useful when you want to create a kernel function that you can
    pass to kernelized models without worrying about parameters each time.
    
    Args:
        kernel_type: string specifying kernel type
                     Options: 'linear', 'polynomial', 'rbf', 'sigmoid'
        **params: kernel-specific parameters
    
    Returns:
        Kernel function that takes (X1, X2) and returns kernel matrix
    
    Examples:
        >>> rbf_func = create_kernel_function('rbf', gamma=0.5)
        >>> K = rbf_func(X_train, X_test)
        
        >>> poly_func = create_kernel_function('polynomial', degree=4, coef0=1.0)
        >>> K = poly_func(X_train, X_train)
    """
    if kernel_type == 'linear':
        # Linear kernel has no parameters
        return lambda X1, X2: linear_kernel(X1, X2)
    
    elif kernel_type == 'polynomial':
        # Extracting polynomial parameters
        degree = params.get('degree', 3)
        coef0 = params.get('coef0', 1.0)
        return lambda X1, X2: polynomial_kernel(X1, X2, degree=degree, coef0=coef0)
    
    elif kernel_type == 'rbf':
        # Extracting gamma parameter
        gamma = params.get('gamma', 0.1)
        return lambda X1, X2: rbf_kernel(X1, X2, gamma=gamma)
    
    elif kernel_type == 'sigmoid':
        # Extracting sigmoid parameters
        gamma = params.get('gamma', 0.1)
        coef0 = params.get('coef0', 0.0)
        return lambda X1, X2: sigmoid_kernel(X1, X2, gamma=gamma, coef0=coef0)
    
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}. "
                        f"Choose from: 'linear', 'polynomial', 'rbf', 'sigmoid'")


def compute_kernel_matrix(X, kernel_func):
    """
    Computes the kernel matrix for a single dataset.
    K[i,j] = kernel(X[i], X[j])
    
    This is useful for visualization and analysis - you can see how similar
    each sample is to every other sample according to the kernel.
    
    Args:
        X: data matrix, shape (n_samples, n_features)
        kernel_func: kernel function to use
    
    Returns:
        Kernel matrix, shape (n_samples, n_samples)
    """
    return kernel_func(X, X)