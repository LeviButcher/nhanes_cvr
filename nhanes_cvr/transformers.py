from functools import reduce
from typing import List, Tuple
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import datasets, cluster, metrics

from nhanes_cvr.utils import XYPair
from imblearn.base import BaseSampler


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
        if (not X is pd.DataFrame):
            X = pd.DataFrame(X)

        if (not y is pd.Series):
            y = pd.Series(y)

        counts = X.count(axis=0)
        total = X.shape[0]
        targetCount = total * self.threshold
        self.colsToKeep = (counts >= targetCount).to_list()
        return self

    def transform(self, X, y=None):
        if (not X is pd.DataFrame):
            X = pd.DataFrame(X)

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
        if (not X is pd.DataFrame):
            X = pd.DataFrame(X)

        if (not y is pd.Series):
            y = pd.Series(y)

        corr = X.corrwith(y).abs()
        self.colsToKeep = (corr >= self.threshold).to_list()

        assert (any(self.colsToKeep)), "At least 1 column has to be selected"
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.iloc[:, self.colsToKeep]


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

    return ((falseX, falseY), (trueX, trueY),)

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


Clusters = Tuple[XYPair, XYPair]


class KMeansUnderSampling(BaseSampler):
    _sampling_type = "under-sampling"

    def __init__(self):
        super().__init__()

    # Need to make this work with the transform
    # Possible need to see how to extend imblearn

    def _fit_resample(self, X, y) -> XYPair:
        if (not X is pd.DataFrame):
            X = pd.DataFrame(X)

        if (not y is pd.Series):
            y = pd.Series(y)

        (majority, minority) = splitByMajority(X, y)

        (minCluster0, minCluster1) = splitByKMeans(minority, 2)
        (majCluster0, majCluster1) = splitByKMeans(majority, 2)

        group1 = combinePairs(minCluster0, majCluster0)
        group2 = combinePairs(minCluster0, majCluster1)
        group3 = combinePairs(minCluster1, majCluster0)
        group4 = combinePairs(minCluster1, majCluster1)

        score1 = silhouetteScore(group1)
        score2 = silhouetteScore(group2)
        score3 = silhouetteScore(group3)
        score4 = silhouetteScore(group4)

        highScore = pd.Series([score1, score2, score3, score4]).idxmax()

        chooseMajority = majCluster0 if (
            highScore == 0 or highScore == 3) else majCluster1

        res = combineAllPairs([minCluster0, minCluster1, chooseMajority])
        print(res)
        return res


def test():
    X, y = datasets.load_breast_cancer(return_X_y=True)

    X = pd.DataFrame(X)
    y = pd.Series(y)

    print(X.shape)
    print(y.shape)

    model = KMeansUnderSampling()
    (newX, newY) = model.fit_resample(X, y)

    print(newX.shape)
    print(newY.shape)

    print(newX.describe())

    exit()
