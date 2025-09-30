import numpy as np


class SVM:
    """
    Support Vector Machine (SVM) for binary classification, implemented from scratch.
    
    Uses gradient descent to minimize hinge loss with L2 regularization.
    The goal is to find a hyperplane that best separates the two classes.
    
    Parameters:
    -----------
    learning_rate : float, default=0.001
        Step size for gradient descent updates
    n_iterations : int, default=1000
        Number of training iterations
    C : float, default=1.0
        Regularization parameter (inverse of lambda)
        Higher C = less regularization, tries to fit training data more closely
        Lower C = more regularization, prefers simpler decision boundary
    verbose : bool, default=False
        Whether to print training progress
    """
    
    def __init__(self, learning_rate=0.001, n_iterations=1000, C=1.0, verbose=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.C = C  # Using C instead of lambda (sklearn convention)
        self.verbose = verbose
        
        # Model parameters - will be learned during training
        self.weights = None
        self.bias = None
        
        # Tracking training progress
        self.history = {
            'loss': [],
            'accuracy': []
        }
    
    def _convert_labels(self, y):
        """
        Converts labels from {0, 1} to {-1, 1} for SVM formulation.
        SVM traditionally uses -1 and +1 labels instead of 0 and 1.
        
        Args:
            y: labels in {0, 1}
        
        Returns:
            Labels in {-1, 1}
        """
        return np.where(y <= 0, -1, 1)
    
    def _compute_hinge_loss(self, X, y):
        """
        Computes the hinge loss with L2 regularization.
        
        Hinge loss formula: max(0, 1 - y*(w^T*x + b))
        This penalizes points that are on the wrong side or too close to the boundary.
        
        Args:
            X: features
            y: labels in {-1, 1}
        
        Returns:
            Total loss value
        """
        n_samples = X.shape[0]
        
        # Computing decision function values: w^T*x + b
        linear_output = np.dot(X, self.weights) + self.bias
        
        # Computing margins: y * (w^T*x + b)
        # If correctly classified with margin > 1, this is positive and large
        # If misclassified or too close, this is small or negative
        margins = y * linear_output
        
        # Hinge loss: max(0, 1 - margin)
        # Loss is 0 if margin >= 1, otherwise it's 1 - margin
        hinge_loss = np.maximum(0, 1 - margins)
        
        # L2 regularization: penalizes large weights
        # Note: In SVM formulation, regularization strength is 1/(2*C)
        regularization_loss = (1 / (2 * self.C)) * np.sum(self.weights ** 2)
        
        # Total loss is average hinge loss plus regularization
        total_loss = np.mean(hinge_loss) + regularization_loss
        
        return total_loss
    
    def fit(self, X, y):
        """
        Trains the SVM using gradient descent on hinge loss.
        
        Args:
            X: training features, shape (n_samples, n_features)
            y: training labels in {0, 1}, shape (n_samples,)
        
        Returns:
            self
        """
        n_samples, n_features = X.shape
        
        # Converting labels from {0, 1} to {-1, 1} for SVM math
        y_converted = self._convert_labels(y)
        
        # Initializing parameters at zero
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Main training loop - gradient descent
        for iteration in range(self.n_iterations):
            # Forward pass: computing predictions and margins
            linear_output = np.dot(X, self.weights) + self.bias
            margins = y_converted * linear_output
            
            # Computing loss to track training progress
            loss = self._compute_hinge_loss(X, y_converted)
            self.history['loss'].append(loss)
            
            # Computing accuracy for monitoring (converting back to {0,1} labels)
            y_pred = self.predict(X)
            accuracy = np.mean(y_pred == y)
            self.history['accuracy'].append(accuracy)
            
            # Computing gradients
            # Key insight: gradient only depends on samples with margin < 1
            # These are called "support vectors"
            condition = margins < 1
            
            # Initializing gradients
            dw = np.zeros(n_features)
            db = 0.0
            
            # Accumulating gradients from misclassified or margin-violating samples
            for i in range(n_samples):
                if condition[i]:
                    # This sample contributes to the gradient
                    dw += -y_converted[i] * X[i]
                    db += -y_converted[i]
            
            # Averaging gradients and adding regularization gradient
            dw = dw / n_samples + (1 / self.C) * self.weights
            db = db / n_samples
            
            # Updating parameters by moving opposite to gradient
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Printing progress periodically
            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        if self.verbose:
            print(f"\nTraining completed!")
            print(f"Final Loss: {self.history['loss'][-1]:.4f}")
            print(f"Final Accuracy: {self.history['accuracy'][-1]:.4f}")
        
        return self
    
    def decision_function(self, X):
        """
        Computes the decision function values: w^T*x + b
        
        The sign of this value determines the predicted class.
        The magnitude indicates confidence (distance from decision boundary).
        
        Args:
            X: features, shape (n_samples, n_features)
        
        Returns:
            Decision values, shape (n_samples,)
        """
        if self.weights is None:
            raise ValueError("Model must be trained before making predictions. Call fit() first.")
        
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X):
        """
        Predicts class labels based on the decision function.
        
        Args:
            X: features, shape (n_samples, n_features)
        
        Returns:
            Predicted labels in {0, 1}, shape (n_samples,)
        """
        # Getting decision function values
        decision_values = self.decision_function(X)
        
        # Converting from {-1, 1} decision to {0, 1} labels
        # If decision >= 0, predict class 1, else predict class 0
        predictions = np.where(decision_values >= 0, 1, 0)
        
        return predictions
    
    def get_params(self):
        """
        Returns the learned parameters and hyperparameters.
        
        Returns:
            Dictionary with all parameters
        """
        return {
            'weights': self.weights,
            'bias': self.bias,
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations,
            'C': self.C
        }


class KernelSVM:
    """
    Kernelized Support Vector Machine using the kernel trick.
    
    Instead of explicitly working with features, this uses kernel functions
    to compute similarities between samples in a high-dimensional space.
    
    Uses gradient descent in dual space with kernel matrix.
    
    Parameters:
    -----------
    kernel_func : callable
        Kernel function that takes two matrices and returns kernel matrix
    learning_rate : float, default=0.001
        Step size for gradient descent
    n_iterations : int, default=1000
        Number of training iterations
    C : float, default=1.0
        Regularization parameter
    verbose : bool, default=False
        Whether to print training progress
    """
    
    def __init__(self, kernel_func, learning_rate=0.001, n_iterations=1000, 
                 C=1.0, verbose=False):
        self.kernel_func = kernel_func
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.C = C
        self.verbose = verbose
        
        # Model parameters in dual space
        self.alpha = None  # Dual coefficients (one per training sample)
        self.X_train = None  # Need to store training data for kernel computations
        self.y_train = None  # Need training labels for predictions
        
        # Training history
        self.history = {
            'loss': [],
            'accuracy': []
        }
    
    def _convert_labels(self, y):
        """Converts labels from {0, 1} to {-1, 1}."""
        return np.where(y <= 0, -1, 1)
    
    def _compute_hinge_loss(self, y, decision_values):
        """
        Computes hinge loss with regularization.
        
        Args:
            y: labels in {-1, 1}
            decision_values: output of decision function
        
        Returns:
            Total loss value
        """
        n_samples = len(y)
        
        # Computing margins
        margins = y * decision_values
        
        # Hinge loss
        hinge_loss = np.maximum(0, 1 - margins)
        
        # Regularization on dual coefficients (alpha)
        regularization_loss = (1 / (2 * self.C)) * np.sum(self.alpha ** 2)
        
        total_loss = np.mean(hinge_loss) + regularization_loss
        
        return total_loss
    
    def fit(self, X, y):
        """
        Trains the kernelized SVM.
        
        The key idea is to work with dual coefficients (alpha) instead of
        weights, and use the kernel matrix to represent similarities.
        
        Args:
            X: training features, shape (n_samples, n_features)
            y: training labels in {0, 1}, shape (n_samples,)
        
        Returns:
            self
        """
        n_samples = X.shape[0]
        
        # Storing training data - needed for computing kernels during prediction
        self.X_train = X.copy()
        self.y_train = self._convert_labels(y)
        
        # Initializing dual coefficients (one per training sample)
        self.alpha = np.zeros(n_samples)
        
        # Computing kernel matrix once at the beginning
        # K[i,j] = kernel(X[i], X[j]) measures similarity between samples i and j
        K = self.kernel_func(X, X)
        
        # Training loop using gradient descent in dual space
        for iteration in range(self.n_iterations):
            # Computing decision values using kernel trick
            # f(x) = Σ(alpha_i * y_i * K(x_i, x))
            decision_values = np.dot((self.alpha * self.y_train), K)
            
            # Computing loss
            loss = self._compute_hinge_loss(self.y_train, decision_values)
            self.history['loss'].append(loss)
            
            # Computing accuracy (converting back to {0, 1} labels)
            y_pred = self.predict(X)
            accuracy = np.mean(y_pred == y)
            self.history['accuracy'].append(accuracy)
            
            # Computing margins to identify support vectors
            margins = self.y_train * decision_values
            
            # Computing gradient with respect to alpha
            # Only samples with margin < 1 contribute to gradient
            condition = margins < 1
            
            # Initializing gradient
            d_alpha = np.zeros(n_samples)
            
            # Accumulating gradient contributions
            for i in range(n_samples):
                if condition[i]:
                    # This sample is a support vector and contributes to gradient
                    d_alpha += -self.y_train[i] * self.y_train * K[:, i]
            
            # Averaging and adding regularization
            d_alpha = d_alpha / n_samples + (1 / self.C) * self.alpha
            
            # Updating dual coefficients
            self.alpha -= self.learning_rate * d_alpha
            
            # Printing progress
            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        if self.verbose:
            print(f"\nTraining completed!")
            print(f"Final Loss: {self.history['loss'][-1]:.4f}")
            print(f"Final Accuracy: {self.history['accuracy'][-1]:.4f}")
            # Counting support vectors (samples with non-zero alpha)
            n_support_vectors = np.sum(np.abs(self.alpha) > 1e-5)
            print(f"Non-zero alphas (support vectors): {n_support_vectors}/{n_samples}")
        
        return self
    
    def decision_function(self, X):
        """
        Computes decision function values using the kernel trick.
        
        Args:
            X: features, shape (n_samples, n_features)
        
        Returns:
            Decision values, shape (n_samples,)
        """
        if self.alpha is None:
            raise ValueError("Model must be trained before making predictions. Call fit() first.")
        
        # Computing kernel between test samples and training samples
        # This is the "trick" - we never explicitly compute feature transformations
        K = self.kernel_func(X, self.X_train)
        
        # Decision function: f(x) = Σ(alpha_i * y_i * K(x_i, x))
        decision_values = np.dot(K, self.alpha * self.y_train)
        
        return decision_values
    
    def predict(self, X):
        """
        Predicts class labels.
        
        Args:
            X: features, shape (n_samples, n_features)
        
        Returns:
            Predicted labels in {0, 1}, shape (n_samples,)
        """
        # Getting decision function values
        decision_values = self.decision_function(X)
        
        # Converting from {-1, 1} decision to {0, 1} labels
        predictions = np.where(decision_values >= 0, 1, 0)
        
        return predictions