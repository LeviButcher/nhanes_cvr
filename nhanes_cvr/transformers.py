from functools import reduce
from typing import Callable, List, Tuple
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import cluster, metrics,  ensemble
from nhanes_cvr.types import XYPair


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
def splitByBinaryClass(X, y) -> Tuple[XYPair]:
    trueLabel = y == 1

    trueX = X.loc[trueLabel, :]
    trueY = y.loc[trueLabel]
    falseX = X.loc[~trueLabel, :]
    falseY = y.loc[~trueLabel]

    return (trueX, trueY), (falseX, falseY)


# Majority always first in tuple
def splitByMajority(X: pd.DataFrame, y: pd.Series):
    ((trueX, trueY), (falseX, falseY)) = splitByBinaryClass(X, y)
    if (trueX.shape[0] > falseX.shape[0]):
        return ((trueX, trueY), (falseX, falseY))

    return ((falseX, falseY), (trueX, trueY))


def splitByClusterMethod(model, data: XYPair, k: int) -> List[XYPair]:
    (X, y) = data
    predictions = model(n_clusters=k).fit_predict(X, y)
    assignedClusters = pd.Series(predictions, index=X.index)
    clusterCounts = assignedClusters.value_counts().sort_values(ascending=False)

    clusterSplits = []
    for i, v in clusterCounts.iteritems():
        toKeep = (assignedClusters == i)
        clusterData = (X.loc[toKeep, :], y.loc[toKeep])
        clusterSplits.append(clusterData)

    return clusterSplits


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


FindBestScore = Callable[[List[float]], int]
KClusterSplitter = Callable[[XYPair, int], List[XYPair]]


def highestScoreIndex(scores: List[float]):
    return pd.Series(scores).idxmax()


def lowestScoreIndex(scores: List[float]):
    return pd.Series(scores).idxmin()


def bestScoreByClosestToMedian(scores: List[float]):
    series = pd.Series(scores)
    med = series.median()
    distances = (series - med).abs()
    return distances.idxmin()


def bestScoreByClosestToMean(scores: List[float]):
    series = pd.Series(scores)
    med = series.mean()
    distances = (series - med).abs()
    return distances.idxmin()


def check_shape(data: XYPair):
    (X, y) = data
    assert X.shape[0] == y.shape[0]
    assert X.index.equals(y.index)
    assert_no_na(X)


def kMeansUnderSampling(X, y, k=2, findBest: FindBestScore = highestScoreIndex, clusterMethod=cluster.KMeans):
    X = transform_to_dataframe(X)
    y = transform_to_series(y)

    (majority, minority) = splitByMajority(X, y)

    minorityClusters = splitByClusterMethod(clusterMethod, minority, k)
    majorityClusters = splitByClusterMethod(clusterMethod, majority, k)

    groups = [(majC, combinePairs(minC, majC))
              for minC in minorityClusters for majC in majorityClusters]

    scores = [silhouetteScore(g) for _, g in groups]

    # Pick majority cluster with best silloute score
    bestIdx = findBest(scores)
    (bestMajority, _) = groups[bestIdx]

    (X, y) = combineAllPairs(minorityClusters + [bestMajority])

    return (X.to_numpy(), y.to_numpy())


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


def iqrRemoval(X, y) -> XYPair:
    X = transform_to_dataframe(X)
    y = transform_to_series(y)

    factor = 1.5
    q3 = X.quantile(q=.75, axis=0)
    q1 = X.quantile(q=.25, axis=0)
    IQR = q3 - q1
    upper_bound = q3 + factor * IQR
    lower_bound = q1 - factor * IQR

    isOutlier = ((X > upper_bound) | (X < lower_bound)).any(axis=1)

    return X.loc[~isOutlier, :], y.loc[~isOutlier]


def iqrBinaryClassesRemoval(X, y) -> XYPair:
    X = transform_to_dataframe(X)
    y = transform_to_series(y)

    ((posX, posY), (negX, negY)) = splitByBinaryClass(X, y)

    (posSet) = iqrRemoval(posX, posY)
    (negSet) = iqrRemoval(negX, negY)

    toKeepX, toKeepY = combinePairs(posSet, negSet)
    # Keep original order
    return X.loc[toKeepX.index, :], y.loc[toKeepY.index]
