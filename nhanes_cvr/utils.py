from typing import Callable, List, Set
import numpy as np
import pandas as pd
from scipy import stats
from nhanes_dl import download, types
from typing import List
import pandas as pd


def getClassName(x) -> str:
    if x is None:
        return "None"

    return x.__class__.__name__


def toUpperCase(l):
    return list(map(lambda l: l.upper(), l))


def yearToMonths(x): return 12 * x


def labelCauseOfDeathAsCVR(nhanse_dataset: pd.DataFrame) -> pd.Series:
    # Different meanings of ucod_leading - https://www.cdc.gov/nchs/data/datalinkage/public-use-2015-linked-mortality-files-data-dictionary.pdf
    # 1 is Diesease of heart
    # 5 is Cerebrovascular Diseases

    # TODO: Encode UCOD_LEADING as a enum
    def isCVR(
        X): return X.UCOD_LEADING == 1 or X.UCOD_LEADING == 5
    # or X.UCOD_LEADING == 7

    # monthsSinceFollowUp = 326 * .5
    # return nhanse_dataset.agg(lambda x: 1 if x.PERMTH_EXM <= monthsSinceFollowUp
    #                           and isCVR(x) else 0, axis=1)

    return nhanse_dataset.agg(isCVR, axis=1)


def remove_outliers(z_score, df, columns=None):
    columns = columns if columns is not None else df.columns
    scores = np.abs(stats.zscore(df.loc[:, columns]))
    return (scores < z_score).all(axis=1)


# TODO: Move this to nhanes_dl
def cache_nhanes(cache_path: str, get_nhanse: Callable[[], types.Codebook], updateCache: bool = False) -> types.Codebook:
    from os.path import exists
    if exists(cache_path) and not updateCache:
        return types.Codebook(pd.read_csv(cache_path))

    else:
        res = get_nhanse()
        res.to_csv(cache_path)
        return res


def generateDownloadConfig(codebooks: List[str]) -> Set[download.CodebookDownload]:
    # NOTE: Could use some general way of getting the suffix for each NHANES Year
    conf = [
        (types.ContinuousNHANES.Fourth, "D"),
        (types.ContinuousNHANES.Fifth, "E"),
        (types.ContinuousNHANES.Sixth, "F"),
        (types.ContinuousNHANES.Seventh, "G"),
        (types.ContinuousNHANES.Eighth, "H"),

        (types.ContinuousNHANES.Ninth, "I"),
        (types.ContinuousNHANES.Tenth, "J"),
    ]

    # May want some way to exclude some codebooks from adding on the suffix
    return {download.CodebookDownload(y, *[f"{c}_{suffix}" for c in codebooks])
            for (y, suffix) in conf}


def doesNHANESNeedRedownloaded(config: Set[download.CodebookDownload]) -> bool:
    codebooks = [c for x in config for c in x.codebooks]
    codebooks.sort()
    newHash = str(codebooks)
    f = open("configHash", "r")
    oldHash = f.read()
    f.close()

    if newHash != oldHash:
        f = open("configHash", "w")
        f.write(newHash)
        f.close()
        return True
    return False
