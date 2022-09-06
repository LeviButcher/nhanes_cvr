from nhanes_cvr.transformers import DropTransformer,  iqrRemoval, kMeansUnderSampling, splitByMajority
import pandas as pd
import numpy as np


def test_dropTransformer_basic():
    # First and Third columns should be dropped
    data = pd.DataFrame([[1, 1,  5],
                         [1, 2,  5],
                         [1, 3,  5],
                         [1, 4,  5],
                         [np.NAN, 5,  5],
                         [np.NAN, 6,  np.NAN],
                         [np.NAN, 7,  np.NAN],
                         [np.NAN, 8,  np.NAN],
                         [np.NAN, 9,  np.NAN],
                         [np.NAN, 10, np.NAN]])

    dt = DropTransformer()
    dt = dt.fit(data)
    res = dt.transform(data)

    assert dt.colsToKeep == [False, True, True]
    assert res.equals(data.iloc[:, 1:])


def test_iqr_removal():
    data = pd.DataFrame([[1, 2],
                         [2, 2],
                         [3, 2],
                         [4, 2],
                         [5, 2],
                         [6, 2],
                         [7, 2],
                         [8, 2],
                         [9, 2],
                         [10, 2]
                         ], columns=["a", "b"])

    Y = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    res = iqrRemoval(data, Y)

    assert res[0].shape <= data.shape


def test_split_by_majority():
    X = pd.DataFrame([[1], [2], [3], [4], [5]])
    Y = pd.Series([0, 0, 1, 1, 1])

    ((majX, majY), (minX, minY)) = splitByMajority(X, Y)

    assert majX.equals(X.loc[2:, :])
    assert majY.equals(Y.loc[2:])

    assert minX.equals(X.loc[:1, :])
    assert minY.equals(Y.loc[:1])


def test_kMeansUnderSampling():
    from sklearn import datasets
    X, Y = datasets.load_iris(return_X_y=True)

    underX, underY = kMeansUnderSampling(X, Y)

    assert underX.shape[0] < X.shape[0]
    assert underY.shape[0] < Y.shape[0]
