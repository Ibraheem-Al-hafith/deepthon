"""
Data Preprocessing Utilities.

This module provides tools for preparing datasets for machine learning, 
including data splitting with support for shuffling and stratification.

Functions:
    train_test_split: Partitions arrays or matrices into random train and test subsets.
"""

import numpy as np
from typing import Any, Tuple, Union, Optional, overload, List

# Type alias for floating or integer NumPy arrays
NDArray = np.ndarray[tuple[Any, ...], np.dtype[Any]]



@overload
def train_test_split(
    x: NDArray, 
    y: None = None, 
    test_size: float = 0.2, 
    stratify: Optional[NDArray] = None, 
    shuffle: bool = True, 
    random_state: Optional[int] = None
) -> Tuple[NDArray, NDArray]: ...

@overload
def train_test_split(
    x: NDArray, 
    y: NDArray, 
    test_size: float = 0.2, 
    stratify: Optional[NDArray] = None, 
    shuffle: bool = True, 
    random_state: Optional[int] = None
) -> Tuple[NDArray, NDArray, NDArray, NDArray]: ...

# deepthon/utils/split.py

def train_test_split(
    x: np.ndarray, 
    y: Optional[np.ndarray] = None, 
    test_size: float = 0.2,
    stratify: Optional[np.ndarray] = None, 
    shuffle: bool = True, 
    random_state: Optional[int] = None
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    
    n_samples = x.shape[0]
    n_test = int(max(1, n_samples * test_size))
    rng = np.random.default_rng(random_state)
    indices = np.arange(n_samples)

    if stratify is not None:
        if stratify.shape[0] != n_samples:
            raise ValueError("Stratify must be the same length as x")
        
        # FIX: Handle 2D (One-Hot) labels for stratification
        strat_labels = stratify
        if stratify.ndim > 1:
            strat_labels = np.argmax(stratify, axis=1)
            
        classes, class_indices = np.unique(strat_labels, return_inverse=True)
        test_indices = []
        
        for i in range(len(classes)):
            c_idx = indices[class_indices == i]
            if shuffle:
                rng.shuffle(c_idx)
            
            n_class_test = int(round(len(c_idx) * test_size))
            test_indices.extend(c_idx[:n_class_test].tolist())
            
        # Ensure exact match for n_test
        if len(test_indices) > n_test:
            test_indices = test_indices[:n_test]
        elif len(test_indices) < n_test:
            remaining = np.setdiff1d(indices, test_indices)
            test_indices.extend(remaining[:n_test - len(test_indices)].tolist())
    else:
        if shuffle:
            rng.shuffle(indices)
        test_indices = indices[:n_test].tolist()

    test_idx_arr = np.array(test_indices)
    train_idx_arr = np.setdiff1d(indices, test_idx_arr)

    if shuffle:
        rng.shuffle(train_idx_arr)
        rng.shuffle(test_idx_arr)

    if y is not None:
        return x[train_idx_arr], x[test_idx_arr], y[train_idx_arr], y[test_idx_arr]
    return x[train_idx_arr], x[test_idx_arr]

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def main() -> None:
    """Demonstrates stratified splitting."""
    # Features 100-109
    x: NDArray = np.arange(100, 110)
    # Targets with class imbalance (60% class 0, 40% class 1)
    y: NDArray = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1])
    
    print("Original Features:", x)
    print("Original Targets: ", y)
    
    # Perform 50/50 split with stratification on y
    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, stratify=y, test_size=0.5
    )

    print("\n--- Split Results ---")
    print("X Train:", xtrain)
    print("Y Train:", ytrain)
    print("X Test: ", xtest)
    print("Y Test: ", ytest)

if __name__ == "__main__":
    main()