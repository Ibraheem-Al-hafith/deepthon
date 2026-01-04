import numpy as np
from typing import Any, Tuple

def train_test_split(x:np.ndarray, y: np.ndarray | None = None, test_size:float = 0.2,
                    stratify: np.ndarray | None = None, shuffle: bool = True,random_state: int|None= None
                    ) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a given data to train and test sets
    
    :param x: Data to be splitted, usually the features data
    :type x: np.ndarray
    :param y: Target data, optional
    :type y: np.ndarray | None
    :param test_size: the proportion of data to be splitted
    :type test_size: float
    :param Stratify: split the data using stratified parts
    :type stratify: np.ndarray | None
    :param shuffle: whether to shuffle the data before splitting, default is true
    :type shffle: bool
    :random_state: the random state for the shuffle
    :type random_state: int | None
    :return: the returned datasets, (train, validation) if y is none, (xtrain, xtest, ytrain, ytes) if both x and y provided
    :rtype: Tuple[ndarray[_AnyShape, dtype[Any]], ndarray[_AnyShape, dtype[Any]]] | Tuple[ndarray[_AnyShape, dtype[Any]], ndarray[_AnyShape, dtype[Any]], ndarray[_AnyShape, dtype[Any]], ndarray[_AnyShape, dtype[Any]]]
    """

    assert    (x.size == y.size if y is not None else False)\
            | (x.size == stratify.size if stratify is not None else False) \
            | (x.size == y.size == stratify.size if (y is not None and stratify is not None) else False) \
            | (y is None and stratify is None) \
            ,"the provided x, y and z must be the same length"
    assert    (stratify.shape in [(len(stratify),), (len(stratify),1)] if stratify is not None else True)\
        ,"stratify must be a one dimenstional array"
    if (not shuffle) and (random_state is not None): raise ValueError("cannot set random state while shuffle is false, remove random state or set shuffle to true")
    indices = np.arange(x.size)
    random_generator = np.random.default_rng(random_state) if random_state is not None else np.random.default_rng()

    test_idx = np.array([])
    if stratify is not None:
        uniques = np.unique(stratify)
        for unq in uniques:
            ind = np.where(stratify == unq)[0]
            if shuffle:
                random_generator.shuffle(ind)
            test_idx = np.append(test_idx,ind[:int(ind.size * test_size)]).astype(int)
            print(ind, test_idx, sep="\n")
    else:
        test_idx = indices[:int(x.size * test_size)]
    train_idx = np.setdiff1d(indices, test_idx)
    if shuffle:
        random_generator.shuffle(train_idx)
        random_generator.shuffle(test_idx)
    if y is not None:
        return x[train_idx], x[test_idx], y[train_idx], y[test_idx]
    return x[train_idx], x[test_idx]


def main():
    x = np.arange(100, 110)
    y = np.array([1,1,1,1,0,0,0,0,0,1])
    print(x)
    print(y)
    for arr in train_test_split(x, y, stratify=y, test_size=0.5):
        print(arr)

if __name__=="__main__":
    main()