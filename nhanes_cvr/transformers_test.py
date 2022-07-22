from nhanes_cvr.transformers import DropTransformer
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
