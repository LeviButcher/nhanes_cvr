import functools
import pandas as pd
from typing import NamedTuple, List, Callable, Any


PostProcess = Callable[[pd.Series], pd.Series]
CombineStrategy = Callable[[pd.DataFrame], pd.Series]


def noProcessing(X: pd.Series) -> pd.Series:
    return X


def firstNonNullCombine(X: pd.DataFrame) -> pd.Series:
    # Just grabs the first non null value from a row
    # Might be a more pandas esc way to do this
    def setAsFirstNonNull(row):
        return functools.reduce(lambda a, b: b if pd.isna(a) else a, row, None)
    return X.agg(setAsFirstNonNull, axis=1)


class CombineFeatures(NamedTuple):
    features: List[str]
    combinedName: str
    combineStrategy: CombineStrategy
    postProcess: PostProcess = noProcessing


def rename(inFeature: str, outName: str, postProcess: PostProcess = noProcessing):
    return CombineFeatures([inFeature], outName, firstNonNullCombine, postProcess)


def create(inFeature: List[str], outName: str, combineStrategy=firstNonNullCombine, postProcess=noProcessing):

    return CombineFeatures(inFeature, outName, combineStrategy, postProcess)


def meanMissingReplacement(X: pd.Series) -> pd.Series:
    return replaceMissingWith(X.mean(), X)


def replaceMissingWith(val: Any, X: pd.Series):
    return X.fillna(val)


def meanCombine(X: pd.DataFrame) -> pd.Series:
    # Should run mean across rows not columns
    # Should handle NAN values
    return X.mean(axis=1)


def runCombine(combine: CombineFeatures, X: pd.DataFrame) -> pd.Series:
    data = X.loc[:, combine.features]
    res = combine.combineStrategy(data)
    res.name = combine.combinedName

    return combine.postProcess(res)


def runCombines(combines: List[CombineFeatures], X: pd.DataFrame) -> pd.DataFrame:
    # Don't keep index, caller should decide if indexing should happen
    return pd.concat([runCombine(c, X) for c in combines], axis=1)
