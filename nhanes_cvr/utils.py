from enum import Enum
from typing import Callable, List, Set, TypeVar
import numpy as np
import pandas as pd
from scipy import stats
from nhanes_dl import download, types
from typing import List
import pandas as pd
from toolz import curry


def getClassName(x) -> str:
    if x is None:
        return "None"

    return x.__class__.__name__


def toUpperCase(l):
    return list(map(lambda l: l.upper(), l))


def yearToMonths(x): return 12 * x

# Different meanings of ucod_leading - https://www.cdc.gov/nchs/data/datalinkage/public-use-2015-linked-mortality-files-data-dictionary.pdf


class LeadingCauseOfDeath(Enum):
    HEART_DISEASE = 1
    MALIGNANT_NEOPLASMS = 2
    CHRONIC_LOWER_RESPIRATORY_DISEASE = 3
    ACCIDENTS = 4
    CEREBROVASCULAR_DISEASE = 5
    ALZHEIMERS_DISEASE = 6
    DIABETES_MELLITUS = 7
    INFLUENZA_PNEUMONIA = 8
    NEPHRITIS_NEPHROTICSYNDROME_NEPHROSIS = 9
    ALL_OTHER_CAUSES = 10


GETY = Callable[[pd.DataFrame], int]


@curry
def labelY(func: GETY, dataset: pd.DataFrame) -> pd.Series:
    return dataset.agg(func, axis=1)


T = TypeVar('T')


def exists(xs: List[T], x: T) -> bool:
    return any([x == y for y in xs])


@curry
def labelCVR(causes: List[LeadingCauseOfDeath], nhanes_dataset: pd.DataFrame) -> pd.Series:
    def toCauseOfDeath(x): return LeadingCauseOfDeath(int(x))
    return labelY(lambda X: 1 if exists(causes, toCauseOfDeath(X.UCOD_LEADING)) else 0, nhanes_dataset)


def removeOutliers(z_score, df, columns=None):
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


def makeDirectoryIfNotExists(directory: str) -> bool:
    import os

    try:
        os.mkdir(directory)
        return True

    except:
        return False
