import numpy as np


class LogisticRegression:
    """
    Logistic Regression for binary classification, implemented from scratch.
    
    Uses gradient descent to minimize binary cross-entropy loss.
    Can include L2 regularization to prevent overfitting.
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        How big of a step to take when updating weights
    n_iterations : int, default=1000
        How many times to go through the training data
    regularization : float, default=0.0
        Strength of L2 regularization (lambda). 0 means no regularization
    verbose : bool, default=False
        Whether to print progress during training
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=0.0, verbose=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.verbose = verbose
        
        # These will be set during training
        self.weights = None
        self.bias = None
        
        # Keeping track of training progress
        self.history = {
            'loss': [],
            'accuracy': []
        }
    
    def _sigmoid(self, z):
        """
        Sigmoid activation function that squashes values between 0 and 1.
        Formula: σ(z) = 1 / (1 + e^(-z))
        
        Args:
            z: input value or array
        
        Returns:
            Output between 0 and 1 (can be interpreted as probability)
        """
        # Clipping z to avoid overflow in the exponential function
        # Very large positive/negative values can cause numerical issues
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_loss(self, y_true, y_pred):
        """
        Computes the binary cross-entropy loss with L2 regularization.
        
        Loss formula: -1/m * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)] + λ/(2m) * ||w||²
        
        Args:
            y_true: actual labels
            y_pred: predicted probabilities
        
        Returns:
            Total loss value (lower is better)
        """
        m = len(y_true)
        
        # Adding small epsilon to avoid log(0) which would give -infinity
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Binary cross-entropy: measures how far predictions are from true labels
        bce_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # L2 regularization penalty: encourages smaller weights
        l2_penalty = (self.regularization / (2 * m)) * np.sum(self.weights ** 2)
        
        total_loss = bce_loss + l2_penalty
        
        return total_loss
    
    def fit(self, X, y):
        """
        Trains the logistic regression model using gradient descent.
        
        Args:
            X: training features, shape (n_samples, n_features)
            y: training labels, shape (n_samples,)
        
        Returns:
            self (allows for method chaining)
        """
        n_samples, n_features = X.shape
        
        # Starting with weights at zero (could also use random initialization)
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Main training loop - gradient descent
        for iteration in range(self.n_iterations):
            # Forward pass: computing predictions
            # Linear combination of inputs
            linear_output = np.dot(X, self.weights) + self.bias
            # Passing through sigmoid to get probabilities
            y_pred = self._sigmoid(linear_output)
            
            # Computing the loss to see how well we're doing
            loss = self._compute_loss(y, y_pred)
            self.history['loss'].append(loss)
            
            # Computing accuracy for monitoring (not used in optimization)
            y_pred_labels = (y_pred >= 0.5).astype(int)
            accuracy = np.mean(y_pred_labels == y)
            self.history['accuracy'].append(accuracy)
            
            # Backward pass: computing gradients
            # Error term: how far off our predictions are
            error = y_pred - y
            
            # Gradient with respect to weights
            # Including the regularization term
            dw = (1 / n_samples) * np.dot(X.T, error) + (self.regularization / n_samples) * self.weights
            
            # Gradient with respect to bias
            db = (1 / n_samples) * np.sum(error)
            
            # Update step: moving weights in opposite direction of gradient
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Printing progress occasionally if verbose is True
            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        if self.verbose:
            print(f"Final Loss: {self.history['loss'][-1]:.4f}")
            print(f"Final Accuracy: {self.history['accuracy'][-1]:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """
        Predicts class probabilities for input samples.
        
        Args:
            X: features, shape (n_samples, n_features)
        
        Returns:
            Predicted probabilities between 0 and 1, shape (n_samples,)
        """
        if self.weights is None:
            raise ValueError("Model must be trained before making predictions. Call fit() first.")
        
        # Computing the linear combination
        linear_output = np.dot(X, self.weights) + self.bias
        # Applying sigmoid to get probabilities
        probabilities = self._sigmoid(linear_output)
        
        return probabilities
    
    def predict(self, X, threshold=0.5):
        """
        Predicts class labels (0 or 1) for input samples.
        
        Args:
            X: features, shape (n_samples, n_features)
            threshold: decision threshold (default=0.5)
        
        Returns:
            Predicted labels (0 or 1), shape (n_samples,)
        """
        # Getting probabilities first
        probabilities = self.predict_proba(X)
        # Converting to binary labels based on threshold
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions
    
    def get_params(self):
        """
        Returns the learned parameters and hyperparameters.
        
        Returns:
            Dictionary containing weights, bias, and hyperparameters
        """
        return {
            'weights': self.weights,
            'bias': self.bias,
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations,
            'regularization': self.regularization
        }


class KernelLogisticRegression:
    """
    Kernelized version of Logistic Regression using the kernel trick.
    
    Instead of learning weights directly, this learns dual coefficients (alpha)
    that work in a higher-dimensional feature space defined by the kernel.
    
    Parameters:
    -----------
    kernel_func : callable
        Kernel function that takes two matrices and returns a kernel matrix
    learning_rate : float, default=0.01
        Step size for gradient descent
    n_iterations : int, default=1000
        Number of training iterations
    regularization : float, default=0.1
        L2 regularization strength
    verbose : bool, default=False
        Whether to print training progress
    """
    
    def __init__(self, kernel_func, learning_rate=0.01, n_iterations=1000, 
                 regularization=0.1, verbose=False):
        self.kernel_func = kernel_func
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.verbose = verbose
        
        # Model parameters - working in dual space
        self.alpha = None  # Dual coefficients (one per training sample)
        self.X_train = None  # Need to store training data for predictions
        
        # Training history
        self.history = {
            'loss': [],
            'accuracy': []
        }
    
    def _sigmoid(self, z):
        """Sigmoid activation function."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_loss(self, y_true, y_pred):
        """Computes binary cross-entropy loss with regularization."""
        m = len(y_true)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Cross-entropy loss
        bce_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # Regularizing the dual coefficients instead of weights
        l2_penalty = (self.regularization / (2 * m)) * np.sum(self.alpha ** 2)
        
        return bce_loss + l2_penalty
    
    def fit(self, X, y):
        """
        Trains the kernelized logistic regression model.
        
        The key difference from standard LR is that we learn dual coefficients
        and work with the kernel matrix instead of explicit features.
        
        Args:
            X: training features, shape (n_samples, n_features)
            y: training labels, shape (n_samples,)
        
        Returns:
            self
        """
        n_samples = X.shape[0]
        
        # Storing training data - we need it for making predictions later
        self.X_train = X.copy()
        
        # Initializing dual coefficients (one per training sample)
        self.alpha = np.zeros(n_samples)
        
        # Computing kernel matrix once at the start
        # K[i,j] = kernel(X[i], X[j]) measures similarity between samples i and j
        K = self.kernel_func(X, X)
        
        # Training loop using gradient descent in dual space
        for iteration in range(self.n_iterations):
            # Forward pass: predictions using kernel trick
            # Instead of w^T·x, we compute Σ(alpha_i * K(x_i, x))
            linear_output = np.dot(K, self.alpha)
            y_pred = self._sigmoid(linear_output)
            
            # Computing loss
            loss = self._compute_loss(y, y_pred)
            self.history['loss'].append(loss)
            
            # Computing accuracy
            y_pred_labels = (y_pred >= 0.5).astype(int)
            accuracy = np.mean(y_pred_labels == y)
            self.history['accuracy'].append(accuracy)
            
            # Computing gradient with respect to alpha coefficients
            error = y_pred - y
            d_alpha = np.dot(K.T, error) / n_samples + (self.regularization / n_samples) * self.alpha
            
            # Updating dual coefficients
            self.alpha -= self.learning_rate * d_alpha
            
            # Printing progress
            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        if self.verbose:
            print(f"\nTraining completed!")
            print(f"Final Loss: {self.history['loss'][-1]:.4f}")
            print(f"Final Accuracy: {self.history['accuracy'][-1]:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """
        Predicts class probabilities using the kernel trick.
        
        Args:
            X: features, shape (n_samples, n_features)
        
        Returns:
            Predicted probabilities, shape (n_samples,)
        """
        if self.alpha is None:
            raise ValueError("Model must be trained before making predictions. Call fit() first.")
        
        # Computing kernel between test samples and training samples
        # This is where the "trick" happens - we never compute explicit features
        K = self.kernel_func(X, self.X_train)
        
        # Making predictions using dual coefficients
        # f(x) = Σ(alpha_i * K(x, x_i))
        linear_output = np.dot(K, self.alpha)
        probabilities = self._sigmoid(linear_output)
        
        return probabilities
    
    def predict(self, X, threshold=0.5):
        """
        Predicts class labels.
        
        Args:
            X: features, shape (n_samples, n_features)
            threshold: classification threshold (default=0.5)
        
        Returns:
            Predicted labels (0 or 1), shape (n_samples,)
        """
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions