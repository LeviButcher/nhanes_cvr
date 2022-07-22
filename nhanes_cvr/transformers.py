from typing import List
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# TODO - WRITE TESTS

class DropTransformer(BaseEstimator, TransformerMixin):
    """
    DropTransformer
    Drops columns that are missing a certain percentage of values
    if threshold is is 0.5 then drop are columns missing data greater then or equal to 0.5 
    """
    threshold: float
    colsToKeep: List[bool]

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        super().__init__()

    def fit(self, X: pd.DataFrame, y=None):
        counts = X.count(axis=0)
        total = X.shape[0]
        targetCount = total * self.threshold
        self.colsToKeep = (counts >= targetCount).to_list()
        return self

    def transform(self, X, y=None):
        return X.iloc[:, self.colsToKeep]


class CorrelationSelection(BaseEstimator, TransformerMixin):
    """
    Correlation Selection

    Selects the features that are greater then or above a correlation threshold to the Y
    """
    threshold: float
    colsToKeep: List[bool]

    def __init__(self, threshold=0.05) -> None:
        self.threshold = threshold
        super().__init__()

    # May need to transform to dataframe
    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        corr = X.corrwith(y).abs()
        self.colsToKeep = (corr >= self.threshold).to_list()

        assert (any(self.colsToKeep)), "At least 1 column has to be selected"
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.iloc[:, self.colsToKeep]
