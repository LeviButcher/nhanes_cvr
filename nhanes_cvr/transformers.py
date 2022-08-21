from functools import reduce
from typing import List, Tuple
from xmlrpc.client import Boolean
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin
from sklearn import datasets, cluster, metrics, linear_model, ensemble
from sklearn.utils import shuffle
from imblearn import pipeline
from nhanes_cvr.utils import XYPair
from imblearn.under_sampling.base import BaseUnderSampler


def transform_to_dataframe(X):
    if (not isinstance(X, pd.DataFrame)):
        return pd.DataFrame(X)
    return X


def transform_to_series(Y):
    if (not isinstance(Y, pd.Series)):
        return pd.Series(Y)
    return Y


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
        X = transform_to_dataframe(X)
        counts = X.count(axis=0)
        total = X.shape[0]
        targetCount = total * self.threshold
        self.colsToKeep = (counts >= targetCount).to_list()
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        return X_.loc[:, self.colsToKeep]


class CorrelationSelection(BaseEstimator, TransformerMixin):
    """
    Correlation Selection

    Selects the features that are greater then or above a correlation threshold to the Y
    """
    threshold: float
    colsToKeep: List[bool]

    def __init__(self, threshold=0.05) -> None:
        self.threshold = threshold
        self.colsToKeep = []
        super().__init__()

    # May need to transform to dataframe
    def fit(self, X, y):
        X = transform_to_dataframe(X)
        y = transform_to_series(y)

        corr = X.corrwith(y).abs()
        self.colsToKeep = (corr >= self.threshold).to_list()

        assert (any(self.colsToKeep)), "At least 1 column has to be selected"
        return self

    def transform(self, X, y=None):
        X = transform_to_dataframe(X)
        X = X.loc[:, self.colsToKeep]
        return X


# Assume 1-0 class label
def splitByBinaryClass(X, y):
    trueLabel = y == 1

    trueX = X.loc[trueLabel, :]
    trueY = y.loc[trueLabel]
    falseX = X.loc[~trueLabel, :]
    falseY = y.loc[~trueLabel]

    return trueX, trueY, falseX, falseY


# Majority always first in tuple
def splitByMajority(X: pd.DataFrame, y: pd.Series):
    trueX, trueY, falseX, falseY = splitByBinaryClass(X, y)
    if (trueX.shape[0] > falseX.shape[0]):
        return ((trueX, trueY), (falseX, falseY))

    return ((falseX, falseY), (trueX, trueY))

# Need to make it easy to combine tuple splits


def splitByKMeans(data: XYPair, k: int) -> List[XYPair]:
    # k should equal length of returned list
    (X, y) = data
    predictions = cluster.KMeans(n_clusters=k).fit_predict(X, y)
    assignedClusters = pd.Series(predictions, index=X.index)
    clusterCounts = assignedClusters.value_counts().sort_values(ascending=False)

    clusterSplits = []
    for i, v in clusterCounts.iteritems():
        toKeep = (assignedClusters == i)
        clusterData = (X.loc[toKeep, :], y.loc[toKeep])
        assert clusterData[0].shape[0] == v
        clusterSplits.append(clusterData)

    return clusterSplits


def combinePairs(pair1: XYPair, pair2: XYPair) -> XYPair:
    (x1, y1) = pair1
    (x2, y2) = pair2

    return (pd.concat([x1, x2]), pd.concat([y1, y2]))


def combineAllPairs(pairList: List[XYPair]) -> XYPair:
    return reduce(combinePairs, pairList)


def silhouetteScore(data: XYPair):
    (X, y) = data
    return metrics.silhouette_score(X, labels=y, random_state=42)


def assert_no_na(X: pd.DataFrame):
    hasNulls = X.isnull().values.any()
    assert (not hasNulls)


def check_shape(data: XYPair):
    (X, y) = data
    # print(X.shape)
    # print(y.shape)
    assert X.shape[0] == y.shape[0]
    assert X.index.equals(y.index)
    assert_no_na(X)


Clusters = Tuple[XYPair, XYPair]


def kMeansUnderSampling(X, y):
    X = transform_to_dataframe(X)
    y = transform_to_series(y)

    check_shape((X, y))
    (majority, minority) = splitByMajority(X, y)
    check_shape(majority)
    check_shape(minority)

    (minCluster0, minCluster1) = splitByKMeans(minority, 2)
    (majCluster0, majCluster1) = splitByKMeans(majority, 2)
    [check_shape(g)
     for g in [minCluster0, minCluster1, majCluster0, majCluster1]]

    group1 = combinePairs(minCluster0, majCluster0)
    group2 = combinePairs(minCluster0, majCluster1)
    group3 = combinePairs(minCluster1, majCluster0)
    group4 = combinePairs(minCluster1, majCluster1)
    [check_shape(g) for g in [group1, group2, group3, group4]]

    score1 = silhouetteScore(group1)
    score2 = silhouetteScore(group2)
    score3 = silhouetteScore(group3)
    score4 = silhouetteScore(group4)

    highScore = pd.Series([score1, score2, score3, score4]).idxmax()

    chooseMajority = majCluster0 if (
        highScore == 0 or highScore == 3) else majCluster1

    res = combineAllPairs([minCluster0, minCluster1, chooseMajority])

    check_shape(res)

    print(res)
    return (shuffle(res[0]), shuffle(res[1]))


class KMeansUnderSampling(BaseUnderSampler):
    def __init__(self):
        super().__init__()

    def _fit_resample(self, X, y) -> XYPair:
        X = transform_to_dataframe(X)
        y = transform_to_series(y)

        check_shape((X, y))
        (majority, minority) = splitByMajority(X, y)
        check_shape(majority)
        check_shape(minority)

        (minCluster0, minCluster1) = splitByKMeans(minority, 2)
        (majCluster0, majCluster1) = splitByKMeans(majority, 2)
        [check_shape(g)
         for g in [minCluster0, minCluster1, majCluster0, majCluster1]]

        group1 = combinePairs(minCluster0, majCluster0)
        group2 = combinePairs(minCluster0, majCluster1)
        group3 = combinePairs(minCluster1, majCluster0)
        group4 = combinePairs(minCluster1, majCluster1)
        [check_shape(g) for g in [group1, group2, group3, group4]]

        score1 = silhouetteScore(group1)
        score2 = silhouetteScore(group2)
        score3 = silhouetteScore(group3)
        score4 = silhouetteScore(group4)

        highScore = pd.Series([score1, score2, score3, score4]).idxmax()

        chooseMajority = majCluster0 if (
            highScore == 0 or highScore == 3) else majCluster1

        res = combineAllPairs([minCluster0, minCluster1, chooseMajority])

        check_shape(res)

        print(res)
        return (shuffle(res[0]), shuffle(res[1]))


class DropNullsTransformer(BaseEstimator, TransformerMixin):
    colsToDrop: List[bool]

    def __init__(self) -> None:
        super().__init__()

    # May need to transform to dataframe
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.colsToDrop = (~X.isnull().any()).tolist()
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        return X.iloc[:, self.colsToDrop]


def outlier_rejection(X, y):
    """This will be our function used to resample our dataset."""
    model = ensemble.IsolationForest(
        max_samples=100, contamination=0.4, random_state=42)
    model.fit(X)
    y_pred = model.predict(X)
    return X[y_pred == 1], y[y_pred == 1]
