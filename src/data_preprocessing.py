import numpy as np
import pandas as pd


def load_data(red_path='data/winequality-red.csv', white_path='data/winequality-white.csv'):
    """
    Loads the red and white wine datasets and combines them into a single DataFrame.
    
    Args:
        red_path: path to red wine CSV file
        white_path: path to white wine CSV file
    
    Returns:
        Combined DataFrame with both wine types
    """
    # Loading both datasets - they use semicolon as separator instead of comma
    red_wine = pd.read_csv(red_path, sep=';')
    white_wine = pd.read_csv(white_path, sep=';')
    
    # Adding a column to distinguish between wine types
    # Using 0 for red wine and 1 for white wine
    red_wine['wine_type'] = 0
    white_wine['wine_type'] = 1
    
    # Combining the datasets vertically (stacking rows)
    wines = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)
    
    print(f"Loaded {len(red_wine)} red wines and {len(white_wine)} white wines")
    print(f"Total dataset size: {len(wines)} samples")
    print(f"\nDataset shape: {wines.shape}")
    print(f"\nColumn names:\n{wines.columns.tolist()}")
    
    return wines


def create_binary_labels(df, threshold=6):
    """
    Converts wine quality scores to binary labels for classification.
    Wines with quality >= 6 are considered "good", others are "bad".
    
    Args:
        df: DataFrame containing a 'quality' column
        threshold: quality score threshold for classification (default=6)
    
    Returns:
        DataFrame with new 'label' column added (1 for good, 0 for bad)
    """
    df = df.copy()
    
    # Creating binary labels based on quality threshold
    # If quality >= 6, the wine is labeled as 1 (good), otherwise 0 (bad)
    df['label'] = (df['quality'] >= threshold).astype(int)
    
    # Calculating and printing the distribution of labels
    print(f"\nBinary label distribution:")
    bad_count = (df['label'] == 0).sum()
    good_count = (df['label'] == 1).sum()
    print(f"Bad wines (quality < {threshold}): {bad_count} ({bad_count/len(df)*100:.1f}%)")
    print(f"Good wines (quality >= {threshold}): {good_count} ({good_count/len(df)*100:.1f}%)")
    
    return df


def preprocess_data(df, test_size=0.2, random_state=42, normalize=True):
    """
    Splits the data into training and test sets, then normalizes the features.
    Uses stratified splitting to maintain class balance in both sets.
    
    Args:
        df: DataFrame with features and 'label' column
        test_size: proportion of data to use for testing (default=0.2 means 20%)
        random_state: random seed for reproducibility
        normalize: whether to apply standardization to features (default=True)
    
    Returns:
        X_train, X_test, y_train, y_test, mean, std
    """
    # Extracting features and labels from the DataFrame
    # We're dropping 'quality' and 'label' columns to keep only the input features
    X = df.drop(['quality', 'label'], axis=1).values
    y = df['label'].values
    
    # Setting random seed so results are reproducible
    np.random.seed(random_state)
    
    # Implementing manual stratified train-test split
    # Stratified means we maintain the same proportion of good/bad wines in both sets
    
    # Finding indices of bad wines (label=0) and good wines (label=1)
    idx_bad = np.where(y == 0)[0]
    idx_good = np.where(y == 1)[0]
    
    # Randomly shuffling the indices within each class
    np.random.shuffle(idx_bad)
    np.random.shuffle(idx_good)
    
    # Calculating how many samples from each class go to the test set
    n_test_bad = int(len(idx_bad) * test_size)
    n_test_good = int(len(idx_good) * test_size)
    
    # Creating test set by taking first n_test samples from each class
    test_idx = np.concatenate([idx_bad[:n_test_bad], idx_good[:n_test_good]])
    # Training set gets the remaining samples
    train_idx = np.concatenate([idx_bad[n_test_bad:], idx_good[n_test_good:]])
    
    # Shuffling train and test indices to mix the classes
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    
    # Using the indices to actually split the data
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    print(f"\nTrain set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    print(f"Train label distribution: Bad={np.sum(y_train==0)}, Good={np.sum(y_train==1)}")
    print(f"Test label distribution: Bad={np.sum(y_test==0)}, Good={np.sum(y_test==1)}")
    
    # Initializing mean and std variables
    mean = None
    std = None
    
    if normalize:
        # Computing mean and standard deviation from the training data only
        # This is crucial - using test data here would cause data leakage
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        
        # Replacing any zero std with 1.0 to avoid division by zero
        # This shouldn't happen with real data but it's a safety check
        std[std == 0] = 1.0
        
        # Applying z-score standardization: (X - mean) / std
        # This transforms features to have mean=0 and std=1
        X_train = (X_train - mean) / std
        # Applying the SAME transformation to test data (using training mean/std)
        X_test = (X_test - mean) / std
        
        print("\nFeatures normalized (standardization: mean=0, std=1)")
    
    return X_train, X_test, y_train, y_test, mean, std


def get_train_val_split(X_train, y_train, val_size=0.2, random_state=42):
    """
    Splits the training data further into training and validation sets.
    This is useful for hyperparameter tuning without touching the test set.
    
    Args:
        X_train: training features
        y_train: training labels
        val_size: proportion to use for validation (default=0.2)
        random_state: random seed
    
    Returns:
        X_train_new, X_val, y_train_new, y_val
    """
    np.random.seed(random_state)
    
    # Using the same stratified split approach as in preprocess_data
    idx_bad = np.where(y_train == 0)[0]
    idx_good = np.where(y_train == 1)[0]
    
    np.random.shuffle(idx_bad)
    np.random.shuffle(idx_good)
    
    # Calculating validation set size for each class
    n_val_bad = int(len(idx_bad) * val_size)
    n_val_good = int(len(idx_good) * val_size)
    
    # Splitting into validation and training indices
    val_idx = np.concatenate([idx_bad[:n_val_bad], idx_good[:n_val_good]])
    train_idx = np.concatenate([idx_bad[n_val_bad:], idx_good[n_val_good:]])
    
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    
    # Creating the actual splits
    X_train_new = X_train[train_idx]
    X_val = X_train[val_idx]
    y_train_new = y_train[train_idx]
    y_val = y_train[val_idx]
    
    print(f"\nTrain-validation split:")
    print(f"New train size: {len(X_train_new)}")
    print(f"Validation size: {len(X_val)}")
    
    return X_train_new, X_val, y_train_new, y_val