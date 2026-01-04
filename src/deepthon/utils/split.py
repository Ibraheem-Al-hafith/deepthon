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

def train_test_split(
    x: np.ndarray, 
    y: Optional[np.ndarray] = None, 
    test_size: float = 0.2,
    stratify: Optional[np.ndarray] = None, 
    shuffle: bool = True, 
    random_state: Optional[int] = None
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Splits datasets into random train and test subsets.

    Provides a utility to partition feature and target arrays. Supports 
    stratification to maintain class proportions in classification tasks.

    Args:
        x (NDArray): Feature data to be split.
        y (Optional[NDArray]): Target data. If provided, returns four arrays.
        test_size (float): The proportion of the dataset to include in the test split.
        stratify (Optional[NDArray]): Array used to perform a stratified split.
        shuffle (bool): Whether or not to shuffle the data before splitting.
        random_state (Optional[int]): Seed for the random number generator.

    Returns:
        Union[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray, NDArray, NDArray]]: 
            Tuple containing (x_train, x_test) or (x_train, x_test, y_train, y_test).

    Raises:
        ValueError: If input lengths are inconsistent or parameters are invalid.
    """
    n_samples = x.shape[0]
    n_test_target = int(n_samples * test_size) # The exact total we need for test
    rng = np.random.default_rng(random_state)
    indices = np.arange(n_samples)

    if stratify is not None:
        # Validate stratify length
        if stratify.shape[0] != n_samples:
            raise ValueError("Stratify must be the same length as x")
            
        classes, class_indices = np.unique(stratify, return_inverse=True)
        test_indices = []
        
        # Calculate how many samples to take from each class
        for i in range(len(classes)):
            c_idx = indices[class_indices == i]
            if shuffle:
                rng.shuffle(c_idx)
            
            # Use rounding that leans toward reaching the target n_test_target
            # We take a proportional slice for each class
            n_class_test = max(1, round(len(c_idx) * test_size))
            test_indices.extend(c_idx[:n_class_test].tolist())
            
        # If rounding created a slight mismatch (e.g., 6 instead of 5), 
        # trim or pad to match n_test_target exactly
        if len(test_indices) > n_test_target:
            test_indices = test_indices[:n_test_target]
        elif len(test_indices) < n_test_target:
            # Add remaining indices that aren't in test yet
            remaining = np.setdiff1d(indices, test_indices)
            test_indices.extend(remaining[:n_test_target - len(test_indices)].tolist())
    else:
        if shuffle:
            rng.shuffle(indices)
        test_indices = indices[:n_test_target].tolist()

    test_idx_arr = np.array(test_indices)
    train_idx_arr = np.setdiff1d(indices, test_idx_arr)

    # Final shuffle to mix classes back together
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