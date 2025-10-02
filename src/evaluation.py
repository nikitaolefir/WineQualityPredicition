import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true, y_pred):
    """
    Calculates common classification metrics from predictions and true labels.
    
    Args:
        y_true: actual labels (numpy array)
        y_pred: predicted labels (numpy array)
    
    Returns:
        Dictionary containing accuracy, precision, recall, F1-score, and confusion matrix values
    """
    # Converting to numpy arrays in case they're lists
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Computing the four values of the confusion matrix
    # TP = correctly predicted positives (predicted 1, actual 1)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    # TN = correctly predicted negatives (predicted 0, actual 0)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    # FP = incorrectly predicted as positive (predicted 1, actual 0)
    fp = np.sum((y_true == 0) & (y_pred == 1))
    # FN = incorrectly predicted as negative (predicted 0, actual 1)
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Computing metrics from confusion matrix values
    # Accuracy = (correct predictions) / (total predictions)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    # Precision = (true positives) / (all predicted positives)
    # Answers: "Of all the wines we predicted as good, how many were actually good?"
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall = (true positives) / (all actual positives)
    # Answers: "Of all the actually good wines, how many did we correctly identify?"
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1-score = harmonic mean of precision and recall
    # It balances both metrics into a single score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Returning all metrics in a dictionary
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
    Prints the metrics in a nicely formatted way.
    
    Args:
        metrics: dictionary from calculate_metrics()
        model_name: name of the model for display purposes
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
    Creates a heatmap visualization of the confusion matrix.
    
    Args:
        y_true: actual labels
        y_pred: predicted labels
        model_name: name for the plot title
        save_path: optional path to save the figure
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Building the confusion matrix manually
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Organizing into a 2x2 matrix
    cm = np.array([[tn, fp], [fn, tp]])
    
    # Creating the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Bad (0)', 'Good (1)'],
                yticklabels=['Bad (0)', 'Good (1)'],
                annot_kws={'size': 16, 'weight': 'bold'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Saving 
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_curves(history, model_name="Model", save_path=None):
    """
    Plots loss and accuracy curves over training epochs.
    
    Args:
        history: dictionary containing 'loss' and optionally 'accuracy' lists
        model_name: name for the plot title
        save_path: optional path to save the figure
    """
    # Creating a list of epoch numbers
    epochs = range(1, len(history['loss']) + 1)
    
    # Checking if accuracy history is available
    has_accuracy = 'accuracy' in history and history['accuracy'] is not None
    
    if has_accuracy:
        # Creating two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left plot: training loss over epochs
        axes[0].plot(epochs, history['loss'], 'b-', linewidth=2, marker='o', markersize=4)
        axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[0].set_title(f'{model_name} - Training Loss', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Right plot: training accuracy over epochs
        axes[1].plot(epochs, history['accuracy'], 'g-', linewidth=2, marker='o', markersize=4)
        axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[1].set_title(f'{model_name} - Training Accuracy', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
    else:
        # Only plotting loss if accuracy is not available
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, history['loss'], 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name} - Training Loss', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def cross_validate(model, X, y, k=5, random_state=42):
    """
    Performs k-fold cross-validation to estimate model performance.
    
    The data is split into k folds, and the model is trained k times,
    each time using a different fold as validation and the rest for training.
    
    Args:
        model: model object with fit() and predict() methods
        X: feature matrix
        y: labels
        k: number of folds (default=5)
        random_state: random seed for reproducibility
    
    Returns:
        Dictionary with fold results, average metrics, and standard deviations
    """
    np.random.seed(random_state)
    
    # Finding indices for each class (stratified CV)
    idx_class_0 = np.where(y == 0)[0]
    idx_class_1 = np.where(y == 1)[0]
    
    # Shuffling indices randomly
    np.random.shuffle(idx_class_0)
    np.random.shuffle(idx_class_1)
    
    # Splitting each class into k equal parts
    folds_0 = np.array_split(idx_class_0, k)
    folds_1 = np.array_split(idx_class_1, k)
    
    # Storing results for each fold
    fold_results = []
    
    print(f"\nPerforming {k}-Fold Cross-Validation...")
    print("="*60)
    
    # Running through each fold
    for fold in range(k):
        print(f"\nFold {fold + 1}/{k}")
        print("-"*40)
        
        # This fold becomes validation set
        val_idx = np.concatenate([folds_0[fold], folds_1[fold]])
        
        # All other folds become training set
        train_idx_0 = np.concatenate([folds_0[i] for i in range(k) if i != fold])
        train_idx_1 = np.concatenate([folds_1[i] for i in range(k) if i != fold])
        train_idx = np.concatenate([train_idx_0, train_idx_1])
        
        # Shuffling train indices to mix classes
        np.random.shuffle(train_idx)
        
        # Creating the actual data splits
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        print(f"Training samples: {len(X_train_fold)}, Validation samples: {len(X_val_fold)}")
        
        # Training the model on this fold
        model.fit(X_train_fold, y_train_fold)
        
        # Getting predictions on validation set
        y_pred_fold = model.predict(X_val_fold)
        
        # Computing metrics for this fold
        metrics = calculate_metrics(y_val_fold, y_pred_fold)
        fold_results.append(metrics)
        
        print(f"Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
    
    # Computing average metrics across all folds
    avg_metrics = {
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'precision': np.mean([r['precision'] for r in fold_results]),
        'recall': np.mean([r['recall'] for r in fold_results]),
        'f1_score': np.mean([r['f1_score'] for r in fold_results]),
    }
    
    # Computing standard deviations to show variability
    std_metrics = {
        'accuracy_std': np.std([r['accuracy'] for r in fold_results]),
        'precision_std': np.std([r['precision'] for r in fold_results]),
        'recall_std': np.std([r['recall'] for r in fold_results]),
        'f1_score_std': np.std([r['f1_score'] for r in fold_results]),
    }
    
    # Printing summary of cross-validation results
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
    Creates a bar chart comparing multiple models on a single metric.
    
    Args:
        results: dictionary mapping model names to their metrics
        metric: which metric to compare ('accuracy', 'precision', 'recall', 'f1_score')
        save_path: optional path to save the figure
    """
    model_names = list(results.keys())
    metric_values = [results[name][metric] for name in model_names]
    
    # Creating a color gradient for the bars
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, metric_values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Adding value labels on top of each bar
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
    Creates a grouped bar chart comparing multiple models across all metrics.
    
    Args:
        results: dictionary mapping model names to their metrics
        save_path: optional path to save the figure
    """
    model_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Setting up bar positions
    x = np.arange(len(model_names))
    width = 0.2  # width of each bar
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Using different colors for each metric
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    # Plotting bars for each metric
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[name][metric] for name in model_names]
        # Each metric gets its own set of bars, offset by i*width
        ax.bar(x + i*width, values, width, label=label, color=colors[i], 
               edgecolor='black', linewidth=1, alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison - All Metrics', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x + width * 1.5)  # centering the labels
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def analyze_misclassifications(X, y_true, y_pred, feature_names=None, n_samples=10):
    """
    Analyzes and displays examples of misclassified samples.
    This helps understand what kinds of wines the model struggles with.
    
    Args:
        X: feature matrix
        y_true: actual labels
        y_pred: predicted labels
        feature_names: list of feature names (optional)
        n_samples: how many misclassified examples to show
    """
    # Finding which samples were misclassified
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    print(f"\n{'='*60}")
    print("Misclassification Analysis")
    print(f"{'='*60}")
    print(f"Total samples: {len(y_true)}")
    print(f"Misclassified: {len(misclassified_idx)} ({len(misclassified_idx)/len(y_true)*100:.2f}%)")
    
    # Breaking down by error type
    false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
    false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]
    
    print(f"\nFalse Positives (predicted Good, actually Bad): {len(false_positives)}")
    print(f"False Negatives (predicted Bad, actually Good): {len(false_negatives)}")
    
    # Showing some example misclassifications
    if len(misclassified_idx) > 0 and n_samples > 0:
        print(f"\n{'='*60}")
        print(f"Sample Misclassified Examples (showing up to {n_samples}):")
        print(f"{'='*60}\n")
        
        # Taking first n_samples misclassified examples
        sample_idx = misclassified_idx[:n_samples]
        
        for i, idx in enumerate(sample_idx):
            print(f"Sample {i+1} (Index: {idx}):")
            print(f"  True Label: {y_true[idx]} ({'Good' if y_true[idx] == 1 else 'Bad'})")
            print(f"  Predicted:  {y_pred[idx]} ({'Good' if y_pred[idx] == 1 else 'Bad'})")
            
            # Showing feature values
            if feature_names:
                print(f"  Features:")
                for j, fname in enumerate(feature_names):
                    print(f"    {fname}: {X[idx][j]:.3f}")
            print()
    
    print(f"{'='*60}\n")