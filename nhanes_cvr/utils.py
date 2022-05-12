from typing import Callable
import numpy as np
import pandas as pd
from scipy import stats
from nhanes_dl import types


def getClassName(x) -> str:
    if x is None:
        return "None"

    return x.__class__.__name__


def toUpperCase(l: list[str]) -> list[str]:
    return list(map(lambda l: l.upper(), l))


def yearToMonths(x): return 12 * x


def labelCauseOfDeathAsCVR(nhanse_dataset: pd.DataFrame) -> pd.Series:
    # Different meanings of ucod_leading - https://www.cdc.gov/nchs/data/datalinkage/public-use-2015-linked-mortality-files-data-dictionary.pdf
    # 1 is Diesease of heart
    # 5 is Cerebrovascular Diseases

    def isCVR(X): return X.UCOD_LEADING == 1 or X.UCOD_LEADING == 5

    monthsSinceFollowUp = 326 * .5
    return nhanse_dataset.agg(lambda x: 1 if x.PERMTH_EXM <= monthsSinceFollowUp
                              and isCVR(x) else 0, axis=1)

    # return nhanse_dataset.agg(isCVR, axis=1)


def remove_outliers(z_score, df, columns=None):
    columns = columns if columns else df.columns
    scores = np.abs(stats.zscore(df.loc[:, columns]))
    return df.loc[(scores < z_score).all(axis=1), :]


# TODO: Move this to nhanes_dl
def cache_nhanes(cache_path: str, get_nhanse: Callable[[], types.Codebook], updateCache: bool = False) -> types.Codebook:
    from os.path import exists
    if exists(cache_path) and not updateCache:
        return types.Codebook(pd.read_csv(cache_path))

    else:
        res = get_nhanse()
        res.to_csv(cache_path)
        return res
