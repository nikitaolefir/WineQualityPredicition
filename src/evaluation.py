import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics: accuracy, precision, recall, F1-score.
    
    Args:
        y_true: True labels (numpy array)
        y_pred: Predicted labels (numpy array)
    
    Returns:
        Dictionary with metrics
    """
    # Convert to numpy arrays if not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }
    
    return metrics


def print_metrics(metrics, model_name="Model"):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary from calculate_metrics()
        model_name: Name of the model for display
    """
    print(f"\n{'='*60}")
    print(f"{model_name} - Performance Metrics")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['tp']}")
    print(f"  True Negatives:  {metrics['tn']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")
    print(f"{'='*60}\n")


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name for the plot title
        save_path: Optional path to save the figure
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate confusion matrix
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    # Create plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Bad (0)', 'Good (1)'],
                yticklabels=['Bad (0)', 'Good (1)'],
                annot_kws={'size': 16, 'weight': 'bold'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_training_curves(history, model_name="Model", save_path=None):
    """
    Plot training curves (loss and accuracy over epochs).
    
    Args:
        history: Dictionary with 'loss' and optionally 'accuracy' lists
                 Example: {'loss': [0.5, 0.4, 0.3], 'accuracy': [0.7, 0.8, 0.85]}
        model_name: Name for the plot title
        save_path: Optional path to save the figure
    """
    epochs = range(1, len(history['loss']) + 1)
    
    # Check if accuracy is available
    has_accuracy = 'accuracy' in history and history['accuracy'] is not None
    
    if has_accuracy:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        axes[0].plot(epochs, history['loss'], 'b-', linewidth=2, marker='o', markersize=4)
        axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[0].set_title(f'{model_name} - Training Loss', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(epochs, history['accuracy'], 'g-', linewidth=2, marker='o', markersize=4)
        axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[1].set_title(f'{model_name} - Training Accuracy', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
    else:
        # Only plot loss
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, history['loss'], 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name} - Training Loss', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def cross_validate(model, X, y, k=5, random_state=42):
    """
    Perform k-fold cross-validation.
    
    Args:
        model: Model object with fit() and predict() methods
        X: Feature matrix (numpy array)
        y: Labels (numpy array)
        k: Number of folds (default: 5)
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with fold results and average metrics
    """
    np.random.seed(random_state)
    
    # Get indices for each class
    idx_class_0 = np.where(y == 0)[0]
    idx_class_1 = np.where(y == 1)[0]
    
    # Shuffle indices
    np.random.shuffle(idx_class_0)
    np.random.shuffle(idx_class_1)
    
    # Split each class into k folds
    folds_0 = np.array_split(idx_class_0, k)
    folds_1 = np.array_split(idx_class_1, k)
    
    # Store results for each fold
    fold_results = []
    
    print(f"\nPerforming {k}-Fold Cross-Validation...")
    print("="*60)
    
    for fold in range(k):
        print(f"\nFold {fold + 1}/{k}")
        print("-"*40)
        
        # Create validation set for this fold
        val_idx = np.concatenate([folds_0[fold], folds_1[fold]])
        
        # Create training set (all other folds)
        train_idx_0 = np.concatenate([folds_0[i] for i in range(k) if i != fold])
        train_idx_1 = np.concatenate([folds_1[i] for i in range(k) if i != fold])
        train_idx = np.concatenate([train_idx_0, train_idx_1])
        
        # Shuffle training indices
        np.random.shuffle(train_idx)
        
        # Split data
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        print(f"Training samples: {len(X_train_fold)}, Validation samples: {len(X_val_fold)}")
        
        # Train model
        model.fit(X_train_fold, y_train_fold)
        
        # Predict on validation set
        y_pred_fold = model.predict(X_val_fold)
        
        # Calculate metrics
        metrics = calculate_metrics(y_val_fold, y_pred_fold)
        fold_results.append(metrics)
        
        print(f"Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
    
    # Calculate average metrics
    avg_metrics = {
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'precision': np.mean([r['precision'] for r in fold_results]),
        'recall': np.mean([r['recall'] for r in fold_results]),
        'f1_score': np.mean([r['f1_score'] for r in fold_results]),
    }
    
    # Calculate standard deviations
    std_metrics = {
        'accuracy_std': np.std([r['accuracy'] for r in fold_results]),
        'precision_std': np.std([r['precision'] for r in fold_results]),
        'recall_std': np.std([r['recall'] for r in fold_results]),
        'f1_score_std': np.std([r['f1_score'] for r in fold_results]),
    }
    
    print("\n" + "="*60)
    print("Cross-Validation Results")
    print("="*60)
    print(f"Average Accuracy:  {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy_std']:.4f}")
    print(f"Average Precision: {avg_metrics['precision']:.4f} ± {std_metrics['precision_std']:.4f}")
    print(f"Average Recall:    {avg_metrics['recall']:.4f} ± {std_metrics['recall_std']:.4f}")
    print(f"Average F1-Score:  {avg_metrics['f1_score']:.4f} ± {std_metrics['f1_score_std']:.4f}")
    print("="*60 + "\n")
    
    return {
        'fold_results': fold_results,
        'avg_metrics': avg_metrics,
        'std_metrics': std_metrics
    }


def plot_model_comparison(results, metric='accuracy', save_path=None):
    """
    Compare multiple models using bar plots.
    
    Args:
        results: Dictionary mapping model names to their metrics
                 Example: {'Model A': {'accuracy': 0.85, ...}, 'Model B': {...}}
        metric: Which metric to compare ('accuracy', 'precision', 'recall', 'f1_score')
        save_path: Optional path to save the figure
    """
    model_names = list(results.keys())
    metric_values = [results[name][metric] for name in model_names]
    
    # Create color map
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, metric_values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    plt.title(f'Model Comparison - {metric.replace("_", " ").title()}', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to {save_path}")
    
    plt.show()


def plot_all_metrics_comparison(results, save_path=None):
    """
    Compare multiple models across all metrics in one plot.
    
    Args:
        results: Dictionary mapping model names to their metrics
        save_path: Optional path to save the figure
    """
    model_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    x = np.arange(len(model_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[name][metric] for name in model_names]
        ax.bar(x + i*width, values, width, label=label, color=colors[i], 
               edgecolor='black', linewidth=1, alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison - All Metrics', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"All metrics comparison saved to {save_path}")
    
    plt.show()


def analyze_misclassifications(X, y_true, y_pred, feature_names=None, n_samples=10):
    """
    Analyze misclassified samples to understand model limitations.
    
    Args:
        X: Feature matrix
        y_true: True labels
        y_pred: Predicted labels
        feature_names: List of feature names (optional)
        n_samples: Number of misclassified samples to display
    """
    # Find misclassified samples
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    print(f"\n{'='*60}")
    print("Misclassification Analysis")
    print(f"{'='*60}")
    print(f"Total samples: {len(y_true)}")
    print(f"Misclassified: {len(misclassified_idx)} ({len(misclassified_idx)/len(y_true)*100:.2f}%)")
    
    # Breakdown by type
    false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
    false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]
    
    print(f"\nFalse Positives (predicted Good, actually Bad): {len(false_positives)}")
    print(f"False Negatives (predicted Bad, actually Good): {len(false_negatives)}")
    
    if len(misclassified_idx) > 0 and n_samples > 0:
        print(f"\n{'='*60}")
        print(f"Sample Misclassified Examples (showing up to {n_samples}):")
        print(f"{'='*60}\n")
        
        sample_idx = misclassified_idx[:n_samples]
        
        for i, idx in enumerate(sample_idx):
            print(f"Sample {i+1} (Index: {idx}):")
            print(f"  True Label: {y_true[idx]} ({'Good' if y_true[idx] == 1 else 'Bad'})")
            print(f"  Predicted:  {y_pred[idx]} ({'Good' if y_pred[idx] == 1 else 'Bad'})")
            
            if feature_names:
                print(f"  Features:")
                for j, fname in enumerate(feature_names):
                    print(f"    {fname}: {X[idx][j]:.3f}")
            print()
    
    print(f"{'='*60}\n")