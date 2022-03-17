import functools
import os
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy import stats


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


CombineDirections = Tuple[List[str], str]


def combine_df_columns(combine_directions: List[CombineDirections], x: pd.DataFrame) -> pd.DataFrame:
    # Do a reduce here where we return the first real value
    def keep_first_non_NAN(x): return functools.reduce(
        lambda b, y: b if not np.isnan(b) else y, x, np.NAN)

    def combine(x: pd.DataFrame, d: CombineDirections):
        cols, target = d
        series = x.loc[:, cols].agg(keep_first_non_NAN, axis=1)
        return pd.DataFrame(series, columns=[target])

    res = [combine(x, d) for d in combine_directions]

    return pd.concat(res, axis=1)


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
