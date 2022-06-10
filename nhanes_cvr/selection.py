import toolz as toolz
from typing import Any, Callable, List, Tuple
import pandas as pd
from nhanes_cvr import utils
import nhanes_cvr.combinefeatures as cf
from toolz import curry

XYPair = Tuple[pd.DataFrame, pd.Series]

Selection = Callable[[pd.DataFrame, pd.Series], XYPair]

# Correlation Selection
# Followed article: https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b


@curry
def correlationSelectionFillNulls(threshold, data: XYPair) -> XYPair:
    X, Y = data
    cor = X.corrwith(Y).abs()
    relevant_features = cor[cor > threshold]
    X = X.loc[:, relevant_features.index.values]
    X = X.fillna(X.mean())

    return (X, Y)


# NOTE: Should nulls be dropped before or after correlation
@curry
def correlationSelection(threshold, data: XYPair) -> XYPair:
    X, Y = data
    cor = X.corrwith(Y).abs()
    relevant_features = cor[cor > threshold]
    X = X.loc[:, relevant_features.index.values]

    return (X, Y)


def fillNullWithMean(data: XYPair) -> XYPair:
    X, Y = data
    X = X.fillna(X.mean())

    return (X, Y)


@curry
def dropSamples(nullThreshold: int, data: XYPair) -> XYPair:
    X, Y = data
    temp = X.assign(Y=Y).dropna(thresh=nullThreshold, axis=0)
    Y = temp.Y
    X = temp.drop(columns=["Y"])
    return (X, Y)


@curry
def handPickedSelection(combineConfig: List[cf.CombineFeatures], data: XYPair) -> XYPair:
    X, Y = data
    X = cf.runCombines(combineConfig, X)
    return (X, Y)


def removeNullSamples(data: XYPair) -> XYPair:
    X, Y = data
    notNull = X.notnull().all(axis=1)
    X = X.loc[notNull, :]
    Y = Y.loc[notNull]
    return (X, Y)


@curry
def removeOutliers(zScore: float, data: XYPair) -> XYPair:
    X, Y = data
    noOutliers = utils.removeOutliers(zScore, X)
    X = X.loc[noOutliers, :]
    Y = Y.loc[noOutliers]
    return (X, Y)
