import functools
import os
from typing import Callable, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from typing import NamedTuple
from nhanes_dl import types
from sklearn.impute import SimpleImputer


def const(x): return x


def getClassName(x) -> str:
    return x.__class__.__name__


ProcessFeatureFunction = Callable[[pd.Series], pd.Series]


class CombineFeatures(NamedTuple):
    inFeatures: List[str]
    feature: str
    scalable: bool = True
    postProcess: ProcessFeatureFunction = const

    # Pure is basically the identity function for AggregateFeature
    # Only 1 Feature in and it becomes the outFeature
    @staticmethod
    def pure(name: str, scalable: bool = True, postProcess: ProcessFeatureFunction = const):
        return CombineFeatures([name], name, scalable, postProcess)

    @staticmethod
    def rename(featureName: str, newFeatureName: str, scalable: bool = True,
               postProcess: ProcessFeatureFunction = const):
        return CombineFeatures([featureName], newFeatureName, scalable, postProcess)


class Experiment(NamedTuple):
    name: str
    featureList: List[CombineFeatures]

# Assume no repeat CombineFeatures
# Should probably remove them


def scalableFeatures(combineDirections: List[CombineFeatures]):
    return [c.feature for c in combineDirections if c.scalable]


def combineExperiments(name: str, experimentList: List[Experiment]) -> Experiment:
    return Experiment(name, [c for e in experimentList for c in e.featureList])


def featureNames(combineDirections: List[CombineFeatures]):
    return [c.feature for c in combineDirections]


def toUpperCase(l):
    return list(map(lambda l: l.upper(), l))


def compose(f, g):
    return lambda x: f(g(x))


def yearToMonths(x): return 12 * x


def labelCauseOfDeathAsCVR(nhanse_dataset: pd.DataFrame):
    # Different meanings of ucod_leading - https://www.cdc.gov/nchs/data/datalinkage/public-use-2015-linked-mortality-files-data-dictionary.pdf
    # 1 is Diesease of heart
    # 5 is Cerebrovascular Diseases

    # monthsSinceFollowUp = 326 * .3

    # return nhanse_dataset.apply(lambda x: 1 if x.PERMTH_EXM <= monthsSinceFollowUp and (x.UCOD_LEADING == 1 or
    #                             x.UCOD_LEADING == 5) else 0, axis=1)

    return nhanse_dataset.apply(lambda x: 1 if (x.UCOD_LEADING == 1 or x.UCOD_LEADING == 5) else 0, axis=1)


def unique(list):
    return functools.reduce(
        lambda l, x: l if x in l else l + [x], list, [])


def ensure_directory_exists(path):
    try:
        os.mkdir(path)
    except:
        pass


def map_dataframe(func, dataframe, columns=None) -> pd.DataFrame:
    columns = columns if columns else dataframe.columns
    r = dataframe.copy()
    r.loc[:, columns] = func(dataframe.loc[:, columns])

    return r


def combine_data_frames(listOfFrames: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    def joinDataFrame(acc, x): return acc.join(
        x[1], how="outer", rsuffix=x[0])

    return functools.reduce(joinDataFrame, listOfFrames, pd.DataFrame())


def remove_outliers(z_score, df, columns=None):
    columns = columns if columns else df.columns
    scores = np.abs(stats.zscore(df.loc[:, columns]))
    return df.loc[(scores < z_score).all(axis=1), :]


def combine_configs(experiment_config):
    variables = unique(
        [x for _, config in experiment_config for x in config])
    return ("combine_all", variables)


def process_combined_features(combine_directions: List[CombineFeatures], x: pd.DataFrame) -> pd.DataFrame:
    # Do a reduce here where we return the first real value
    def keep_first_non_NAN(x): return functools.reduce(
        lambda b, y: y if y != np.NAN else b, x, np.NAN)

    def combine(d: CombineFeatures, x: pd.DataFrame):

        series = x.loc[:, d.inFeatures].agg(keep_first_non_NAN, axis=1)
        series.name = d.feature
        return d.postProcess(series)

    res = [combine(d, x) for d in combine_directions]

    return pd.concat(res, axis=1)


def process_dataset(dataset: pd.DataFrame, combineDirections: List[CombineFeatures], labelFunc):
    dataset = process_combined_features(
        combineDirections, dataset).assign(Y=labelFunc(dataset))

    dataset = dataset.dropna()

    Y = dataset.Y
    X = dataset.drop(columns="Y")

    return X, Y


def meanReplacement(x: pd.Series) -> pd.Series:
    res = SimpleImputer().fit_transform(pd.DataFrame(x))

    x[:] = res[:, 0]  # EWWW
    return x


def answeredYesOnQuestion(x: pd.Series) -> pd.Series:
    return x.map(lambda d: 1 if d == 1 else 0)


# Need to include this in nhanes-dl
def cache_nhanes(cache_path: str, get_nhanse: Callable[[], types.Codebook]) -> types.Codebook:
    from os.path import exists
    if exists(cache_path):
        return types.Codebook(pd.read_csv(cache_path))

    else:
        res = get_nhanse()
        res.to_csv(cache_path)
        return res
