from enum import Enum
from typing import Callable, List, Set, Tuple, TypeVar
import numpy as np
import pandas as pd
from pygments import highlight
from scipy import stats
from nhanes_dl import download, types
from typing import List
import pandas as pd
from toolz import curry

from nhanes_cvr.mlProcess import XYPair


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


def toCauseOfDeath(x): return LeadingCauseOfDeath(int(x))


def nhanesToQuestionnaireSet(nhanes_dataset) -> Tuple[pd.DataFrame, pd.Series]:
    X = nhanes_dataset.drop(columns=["MCQ160F", "MCQ160C", "MCQ160B", "MCQ160E"]) \
        .drop(columns=download.getMortalityColumns())
    Y = labelViaQuestionnaire(nhanes_dataset)
    return (X, Y)


def nhanesToMortalitySet(nhanes_dataset) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = nhanes_dataset.loc[nhanes_dataset.ELIGSTAT == 1, :]
    dataset = dataset.loc[dataset.MORTSTAT == 1, :]
    X = dataset.drop(columns=download.getMortalityColumns())
    normalCVRDeath = [LeadingCauseOfDeath.HEART_DISEASE,
                      LeadingCauseOfDeath.CEREBROVASCULAR_DISEASE]
    Y = labelCVR(normalCVRDeath, dataset)  # type: ignore
    return (X, Y)  # type: ignore


@curry
def nhanesToMortalityWithinTimeSet(withinYear, nhanes_dataset) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = nhanes_dataset.loc[nhanes_dataset.ELIGSTAT == 1, :]
    dataset = dataset.loc[dataset.MORTSTAT == 1, :]
    X = dataset.drop(columns=download.getMortalityColumns())
    Y = labelCVRDeathWithinTime(withinYear, dataset)
    return (X, Y)  # type: ignore


def labelCVRBasedOnCardiovascularCodebook(nhanes_dataset) -> XYPair:
    dataset = nhanes_dataset.drop(columns=download.getMortalityColumns())
    discomfortInChest = (dataset.CDQ001 == 1)
    shortnessOfBreath = (dataset.CDQ010 == 1)
    Y = (discomfortInChest & shortnessOfBreath).astype(int)
    X = dataset.drop(columns=["CDQ001", "CDQ010"])
    return (X, Y)


@curry
def labelCVRBasedOnLabMetrics(threshold: int, nhanes_dataset: pd.DataFrame) -> XYPair:
    dataset = nhanes_dataset.drop(
        columns=download.getMortalityColumns())  # type: ignore
    highChol = (dataset.LBXTC > 239)
    highLDL = (dataset.LBDLDL > 130)
    lowHDL = (dataset.LBDHDD < 40)
    highTG = (dataset.LBXTR > 200)
    highGlucose = (dataset.LBXGLU > 200)
    cvRisk = pd.concat(
        [highChol, highLDL, lowHDL, highGlucose, highTG], axis=1).astype(int).sum(axis=1)
    Y = (cvRisk > threshold).astype(int)
    X = dataset.drop(columns=["LBXTC", "LBDLDL", "LBDHDD", "LBXGLU", "LBXTR"])
    return (X, Y)


def labelViaQuestionnaire(nhanes_dataset) -> pd.Series:

    def convertQuestionnaire(X):
        if X.MCQ160F == 1:  # Had Stroke
            return 1
        elif X.MCQ160C == 1:  # Had Coronary heart disease
            return 1
        elif X.MCQ160B == 1:  # Had Congestive Heart Failure
            return 1
        elif X.MCQ160E == 1:  # Had Heart Attack
            return 1
        else:
            return 0

    return labelY(convertQuestionnaire, nhanes_dataset)  # type: ignore


def labelCVRDeathWithinTime(withinYear: int, dataset: pd.DataFrame) -> pd.Series:
    cvdDeath = (dataset.UCOD_LEADING == 1) | (dataset.UCOD_LEADING == 5)
    diedWithinYear = (dataset.PERMTH_INT / 12) <= withinYear
    return (cvdDeath & diedWithinYear).astype(int)


@curry
def labelCVR(causes: List[LeadingCauseOfDeath], nhanes_dataset: pd.DataFrame) -> pd.Series:

    return labelY(lambda X: 1 if exists(causes, toCauseOfDeath(X.UCOD_LEADING))
                  else 0, nhanes_dataset)  # type: ignore


def labelCVRAndDiabetes(nhanes_dataset: pd.DataFrame) -> pd.Series:
    def labelFunc(X: pd.DataFrame) -> int:
        cause = toCauseOfDeath(X.UCOD_LEADING)
        if exists([LeadingCauseOfDeath.HEART_DISEASE, LeadingCauseOfDeath.CEREBROVASCULAR_DISEASE], cause):
            return 1
        elif cause == LeadingCauseOfDeath.DIABETES_MELLITUS:
            return 2
        return 0

    return labelY(labelFunc, nhanes_dataset)  # type: ignore


def removeOutliers(z_score, df, columns=None):
    columns = columns if columns is not None else df.columns
    scores = np.abs(stats.zscore(df.loc[:, columns]))
    return (scores < z_score).all(axis=1)


# TODO: Move this to nhanes_dl
def cache_nhanes(cache_path: str, get_nhanse: Callable[[], types.Codebook], updateCache: bool = False) -> types.Codebook:
    from os.path import exists
    if exists(cache_path) and not updateCache:
        return types.Codebook(pd.read_csv(cache_path, low_memory=False))

    else:
        res = get_nhanse()
        res.to_csv(cache_path)
        return res


def generateDownloadConfig(codebooks: List[str]) -> Set[download.CodebookDownload]:
    # NOTE: Could use some general way of getting the suffix for each NHANES Year
    conf = [
        # (types.ContinuousNHANES.Fourth, "D"),
        (types.ContinuousNHANES.Fifth, "E"),
        (types.ContinuousNHANES.Sixth, "F"),
        (types.ContinuousNHANES.Seventh, "G"),
        (types.ContinuousNHANES.Eighth, "H"),

        # (types.ContinuousNHANES.Ninth, "I"),
        # (types.ContinuousNHANES.Tenth, "J"),
    ]

    # May want some way to exclude some codebooks from adding on the suffix
    return {download.CodebookDownload(y, *[f"{c}_{suffix}" for c in codebooks])
            for (y, suffix) in conf}


def doesNHANESNeedDownloaded(config: Set[download.CodebookDownload]) -> bool:
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
