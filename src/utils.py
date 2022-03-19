import functools
import os
from typing import Callable, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import NamedTuple


def const(x): return x


ProcessFeatureFunction = Callable[[pd.Series], pd.Series]


class CombineFeatures(NamedTuple):
    inFeatures: List[str]
    feature: str
    postProcess: ProcessFeatureFunction = const

    # Pure is basically the identity function for AggregateFeature
    # Only 1 Feature in and it becomes the outFeature
    def pure(name: str, postProcess=const):
        return CombineFeatures([name], name, postProcess)

    def rename(featureName: str, newFeatureName: str, postProcess=const):
        return CombineFeatures([featureName], newFeatureName, postProcess)


class Experiment(NamedTuple):
    name: str
    featureList: List[CombineFeatures]

# Assume no repeat CombineFeatures
# Should probably remove them


def combineExperiments(name: str, experimentList: List[Experiment]) -> Experiment:
    return Experiment(name, [c for e in experimentList for c in e.featureList])


def featureNames(combineDirections: List[CombineFeatures]):
    return [c.feature for c in combineDirections]


def toUpperCase(l):
    return list(map(lambda l: l.upper(), l))


def compose(f, g):
    return lambda x: f(g(x))


def labelCauseOfDeathAsCVR(nhanse_dataset):
    # Different meanings of ucod_leading - https://www.cdc.gov/nchs/data/datalinkage/public-use-2015-linked-mortality-files-data-dictionary.pdf
    # 1 is Diesease of heart
    # 5 is Cerebrovascular Diseases

    return nhanse_dataset.apply(lambda x: 1 if x.UCOD_LEADING == 1 or
                                x.UCOD_LEADING == 5 else 0, axis=1)


def unique(list):
    return functools.reduce(
        lambda l, x: l if x in l else l + [x], list, [])


def ensure_directory_exists(path):
    try:
        os.mkdir(path)
    except:
        pass


def map_dataframe(func, dataframe) -> pd.DataFrame:
    r = dataframe.copy()
    r.loc[:, :] = func(dataframe)

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
        lambda b, y: b if not np.isnan(b) else y, x, np.NAN)

    def combine(d: CombineFeatures, x: pd.DataFrame):
        (inFeatures, featureName, postProcess) = d
        series = x.loc[:, inFeatures].agg(keep_first_non_NAN, axis=1)
        return postProcess(pd.DataFrame(series, columns=[featureName]))

    res = [combine(d, x) for d in combine_directions]

    return pd.concat(res, axis=1)


def process_dataset(dataset: pd.DataFrame, combineDirections: List[CombineFeatures], labelFunc):
    dataset = process_combined_features(
        combineDirections, dataset).assign(Y=labelFunc(dataset))

    dataset = dataset.dropna()

    Y = dataset.Y
    X = dataset.drop(columns="Y")

    return X, Y
