from functools import reduce
from typing import Callable, List, Tuple, Union
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


class DropNullsTransformer(BaseEstimator, TransformerMixin):
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


class DropTransformer(BaseEstimator, TransformerMixin):
    """
    DropTransformer
    Drops columns that are missing a certain percentage of values
    if threshold is is 0.5 then drop are columns missing data greater then or equal to 0.5
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        super().__init__()

    def fit(self, X, y=None):
        X = transform_to_dataframe(X)
        counts = X.count(axis=0)
        total = X.shape[0]
        targetCount = total * self.threshold
        self.colsToKeep = (counts >= targetCount).to_list()
        return self

    def transform(self, X, y=None):
        X = transform_to_dataframe(X)
        return X.loc[:, self.colsToKeep]


def correlationScore(X, y):
    X = transform_to_dataframe(X)
    y = transform_to_series(y)
    return X.corrwith(y).abs()


# Assume 1-0 class label
def splitByBinaryClass(X: pd.DataFrame, y: pd.Series) -> Tuple[XYPair, XYPair]:
    trueLabel = y == 1

    trueX = X.loc[trueLabel, :]
    trueY = y.loc[trueLabel]
    falseX = X.loc[~trueLabel, :]
    falseY = y.loc[~trueLabel]

    true = (trueX, trueY)
    false = (falseX, falseY)

    return (true, false)


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
    for i, v in clusterCounts.iteritems():  # type: ignore
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
    for i, v in clusterCounts.iteritems():  # type: ignore
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


FindBestScore = Callable[[List[float]], Union[int, str]]
KClusterSplitter = Callable[[XYPair, int], List[XYPair]]


def highestScoreIndex(scores: List[float]) -> int | str:
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

    # Pick majority cluster with best silhouette score
    bestIdx = findBest(scores)  # type: ignore
    (bestMajority, _) = groups[bestIdx]  # type: ignore

    (X, y) = combineAllPairs(minorityClusters + [bestMajority])

    return (X.to_numpy(), y.to_numpy())


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


def handpickedSelector(X, y=None) -> XYPair:
    # NEED to use sklearn-pandas to make this work
    X = transform_to_dataframe(X)
    y = transform_to_series(y)
    demoCols = ["RIDAGEYR", "RIAGENDR", "RIDRETH1"]
    labCols = ["LBXTC", "LBDLDL", "LBXTR", "LBDHDD", "LBXGLU",
               "LBXIN", "LBXHGB", "LBXGH", "LBXAPB", "URXUIO", "URXUCR"]
    questionnaireCols = ["CDQ001", "CDQ010", "SMQ020", "MCQ010"]
    allCols = demoCols + labCols + questionnaireCols
    availableCols = X.columns.intersection(allCols).to_list()
    X = X.loc[:, availableCols]

    return X, y

    # # Demographics
    # cf.rename("RIDAGEYR", "AGE"),
    # cf.rename("RIAGENDR", "GENDER"),
    # cf.rename("RIDRETH1", "Race"),

    # # Lab
    # cf.rename("LBXTC", "Total_Chol",
    #           postProcess=cf.meanMissingReplacement),
    # cf.rename("LBDLDL", "LDL", postProcess=cf.meanMissingReplacement),
    # cf.rename("LBXTR", "TG", postProcess=cf.meanMissingReplacement),
    # cf.rename("LBDHDD", "HDL", postProcess=cf.meanMissingReplacement),

    # cf.create(["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"], "SYSTOLIC",
    #           combineStrategy=cf.meanCombine, postProcess=cf.meanMissingReplacement),
    # cf.create(["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"], "DIASTOLIC",
    #           combineStrategy=cf.meanCombine, postProcess=cf.meanMissingReplacement),

    # cf.rename("LBXGLU", "FBG", postProcess=cf.meanMissingReplacement),
    # cf.rename("LBXIN", "INSULIN", postProcess=cf.meanMissingReplacement),

    # cf.rename("LBXHGB", "HEMOGOBLIN",
    #           postProcess=cf.meanMissingReplacement),
    # cf.rename("LBXGH", "GLYCOHEMOGLOBIN",
    #           postProcess=cf.meanMissingReplacement),

    # # Very few have this
    # # cf.rename("LBXAPB", "APOLIPOPROTEIN",
    # #           postProcess=cf.meanMissingReplacement),

    # # # Takes away like 700 samples
    # # cf.rename("URXUIO", "IODINE",
    # #           postProcess=cf.meanMissingReplacement),
    # # cf.rename("URXUCR", "CREATINE",
    # #           postProcess=cf.meanMissingReplacement),

    # # # Questionaire
    # cf.rename("CDQ001", "CHEST_PAIN", postProcess=standardYesNoProcessor),
    # cf.rename("CDQ010", "SHORTNESS_OF_BREATHS",
    #           postProcess=standardYesNoProcessor),
    # # # # Could add more from CDQ
    # cf.rename("SMQ020", "SMOKED_AT_LEAST_100_IN_LIFE",
    #           postProcess=standardYesNoProcessor),

    # # # Might add Sleep, Weight History,
    # cf.rename("MCQ010", "TOLD_HAVE_ASTHMA",
    #           postProcess=standardYesNoProcessor)
