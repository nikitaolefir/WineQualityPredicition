# Wine Quality Prediction

Binary classification of wine quality using Support Vector Machines and Logistic Regression implemented from scratch.

## Project Overview

This project predicts wine quality (good ≥ 6 vs bad < 6) based on physicochemical properties using machine learning algorithms implemented without scikit-learn model classes.

**Best Model Performance:** 82.31% F1-score, 76.19% accuracy (Kernel Logistic Regression with RBF kernel)

## Project Structure

```
WINEQUALITYPREDICTION/
│
├── data/                           # Dataset files
│   ├── winequality-red.csv        # Red wine data (1,599 samples)
│   └── winequality-white.csv      # White wine data (4,898 samples)
│
├── src/                            # Source code (all from scratch)
│   ├── __init__.py                # Package initialization
│   ├── data_preprocessing.py      # Data loading, splitting, normalization
│   ├── evaluation.py              # Metrics, cross-validation, visualization
│   ├── kernels.py                 # Kernel functions (linear, polynomial, RBF)
│   ├── logistic_regression.py     # LR and Kernel LR implementations
│   └── svm.py                     # SVM and Kernel SVM implementations
│
├── results/                        # Generated visualizations
│   ├── *_confusion_matrix.png     # Confusion matrices for all models
│   ├── *_training_curves.png      # Loss and accuracy curves
│   └── model_comparison.png       # Performance comparison chart
│
├── side_notebooks/                # Exploratory analysis
│   ├── exploration.ipynb          # Data exploration and visualization
│   └── experiments.ipynb          # Hyperparameter tuning experiments
│
├── main.ipynb                      # Main execution notebook
└── README.md
└── WineQualityPredictionReport.pdf  #report with key findings                  
```

## Implementation Details

### Models Implemented (From Scratch)

1. **Logistic Regression**
   - Binary cross-entropy loss with L2 regularization
   - Gradient descent optimization
   - Sigmoid activation function

2. **Support Vector Machine**
   - Hinge loss with L2 regularization
   - Gradient descent in primal space
   - Support vector identification

3. **Kernel Logistic Regression**
   - Dual formulation with kernel trick
   - Supports polynomial and RBF kernels

4. **Kernel SVM**
   - Dual formulation with kernel trick
   - Supports polynomial and RBF kernels

### Kernel Functions

- **Linear:** K(x, y) = x^T · y
- **Polynomial:** K(x, y) = (x^T · y + c)^d
- **RBF:** K(x, y) = exp(-γ ||x - y||²)

## Dataset

- **Source:** UCI Machine Learning Repository
- **Total samples:** 6,497 wines (1,599 red + 4,898 white)
- **Features:** 11 physicochemical properties + wine type
- **Target:** Binary (good ≥ 6 vs bad < 6)
- **Class distribution:** 63.3% good, 36.7% bad

## Methodology

1. **Data Preprocessing:**
   - Stratified 80-20 train-test split
   - Z-score normalization (using training statistics only)
   - No data leakage

2. **Hyperparameter Tuning:**
   - 5-fold stratified cross-validation
   - Grid search for regularization (λ) and kernel parameters (d, γ)

3. **Evaluation Metrics:**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion matrices
   - Training curves
   - Misclassification analysis

## Requirements

```
Python 3.8+
numpy
pandas
matplotlib
seaborn
```

## Results Summary

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | 75.65% | 81.61% |
| SVM | 69.34% | 76.64% |
| **Kernel LR (RBF, γ=0.1)** | **76.19%** | **82.31%** |
| Kernel LR (RBF, γ=0.5) | 74.19% | 81.46% |
| Kernel SVM (RBF, γ=0.1) | 68.03% | 78.22% |

## Key Findings

1. Logistic Regression outperformed SVM (6.3% accuracy advantage)
2. RBF kernels significantly better than polynomial kernels
3. Lower gamma (0.1) generalized better than higher values
4. Model exhibits 2:1 false positive to false negative ratio
5. Training time: LR (0.16s) vs SVM (4.91s) vs Kernel SVM (34-145s)

## Notes

- All random seeds set to 42 for reproducibility
- Models track training history (loss, accuracy) for analysis
- Kernel models store training data for predictions
- Stratified sampling maintains class balance throughout