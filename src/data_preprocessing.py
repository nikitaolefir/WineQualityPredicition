import numpy as np
import pandas as pd

def load_data(red_path='data/raw/winequality-red.csv', white_path='data/raw/winequality-white.csv'):
    """
    Load and combine red and white wine datasets.
    
    Args:
        red_path: Path to red wine CSV
        white_path: Path to white wine CSV
    
    Returns:
        Combined DataFrame with all wines
    """
    red_wine = pd.read_csv(red_path, sep=';')
    white_wine = pd.read_csv(white_path, sep=';')
    
    red_wine['wine_type'] = 0  # Red wine
    white_wine['wine_type'] = 1  # White wine
    
    wines = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)
    
    print(f"Loaded {len(red_wine)} red wines and {len(white_wine)} white wines")
    print(f"Total dataset size: {len(wines)} samples")
    print(f"\nDataset shape: {wines.shape}")
    
    return wines


def create_binary_labels(df, threshold=6):
    """
    Function converts wine quality to binary labels: good (≥6) vs bad (<6).
    
    Args:
        df: DataFrame with 'quality' column
        threshold: Quality threshold (default: 6)
    
    Returns:
        DataFrame with added 'label' column (1=good, 0=bad)
    """
    df = df.copy()
    df['label'] = (df['quality'] >= threshold).astype(int)
    
    print(f"\nBinary label distribution:")
    print(f"Bad wines (quality < {threshold}): {(df['label'] == 0).sum()} ({(df['label'] == 0).sum()/len(df)*100:.1f}%)")
    print(f"Good wines (quality >= {threshold}): {(df['label'] == 1).sum()} ({(df['label'] == 1).sum()/len(df)*100:.1f}%)")
    
    return df


def preprocess_data(df, test_size=0.2, random_state=42, normalize=True):
    """
    Preprocess data: split into train/test and normalize.
    
    Args:
        df: DataFrame with features and 'label' column
        test_size: Proportion of test set (default: 0.2)
        random_state: Random seed for reproducibility
        normalize: Whether to normalize features (default: True)
    
    Returns:
        X_train, X_test, y_train, y_test
        mean, std (if normalize=True, else None, None)
    """
    # Separate features and labels
    X = df.drop(['quality', 'label'], axis=1).values
    y = df['label'].values
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Stratified train-test split manually
    # Get indices for each class
    idx_class_0 = np.where(y == 0)[0]
    idx_class_1 = np.where(y == 1)[0]
    
    # Shuffle indices
    np.random.shuffle(idx_class_0)
    np.random.shuffle(idx_class_1)
    
    # Split each class
    n_test_0 = int(len(idx_class_0) * test_size)
    n_test_1 = int(len(idx_class_1) * test_size)
    
    test_idx = np.concatenate([idx_class_0[:n_test_0], idx_class_1[:n_test_1]])
    train_idx = np.concatenate([idx_class_0[n_test_0:], idx_class_1[n_test_1:]])
    
    # Shuffle train and test indices
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    
    # Create train and test sets
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"\nTrain set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    print(f"Train label distribution: Bad={np.sum(y_train==0)}, Good={np.sum(y_train==1)}")
    print(f"Test label distribution: Bad={np.sum(y_test==0)}, Good={np.sum(y_test==1)}")
    
    # Normalize features (standardization: mean=0, std=1)
    mean, std = None, None
    if normalize:
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        
        # Avoid division by zero
        std[std == 0] = 1.0
        
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        print("\nFeatures normalized (standardization: mean=0, std=1)")
    
    return X_train, X_test, y_train, y_test, mean, std


def get_train_val_split(X_train, y_train, val_size=0.2, random_state=42):
    """
    Split training data into train and validation sets (useful for hyperparameter tuning).
    
    Args:
        X_train: Training features
        y_train: Training labels
        val_size: Proportion of validation set
        random_state: Random seed
    
    Returns:
        X_train_new, X_val, y_train_new, y_val
    """
    np.random.seed(random_state)
    
    # Stratified split
    idx_class_0 = np.where(y_train == 0)[0]
    idx_class_1 = np.where(y_train == 1)[0]
    
    np.random.shuffle(idx_class_0)
    np.random.shuffle(idx_class_1)
    
    n_val_0 = int(len(idx_class_0) * val_size)
    n_val_1 = int(len(idx_class_1) * val_size)
    
    val_idx = np.concatenate([idx_class_0[:n_val_0], idx_class_1[:n_val_1]])
    train_idx = np.concatenate([idx_class_0[n_val_0:], idx_class_1[n_val_1:]])
    
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    
    X_train_new = X_train[train_idx]
    X_val = X_train[val_idx]
    y_train_new = y_train[train_idx]
    y_val = y_train[val_idx]
    
    print(f"\nTrain-validation split:")
    print(f"New train size: {len(X_train_new)}")
    print(f"Validation size: {len(X_val)}")
    
    return X_train_new, X_val, y_train_new, y_val