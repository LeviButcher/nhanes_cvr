import pandas as pd
import numpy as np
from nhanes_cvr.selection import dropColumns


def test_dropColumns():
    data = pd.DataFrame({"a": [1, 2, np.NAN, np.NAN], "b": [1, 2, 3, 4]})
    Y = pd.Series([1, 2, 3, 4])
    (X, newY) = dropColumns(.50, (data, Y))  # type: ignore

    assert X.equals(data.drop(columns=["a"]))
    assert newY.equals(Y)


def test_dropColumns_complex():
    data = pd.DataFrame({"a": [1, 2, np.NAN, np.NAN],
                         "b": [np.NAN, 2, 3, 4],
                         "c": [np.NAN, np.NAN, np.NAN, np.NAN]})
    Y = pd.Series([1, 2, 3, 4])
    (X, newY) = dropColumns(.50, (data, Y))  # type: ignore

    assert X.equals(data.drop(columns=["a", "c"]))
    assert newY.equals(Y)
