from enum import Enum
from functools import reduce
from typing import List, Tuple, TypeVar
import numpy as np
import pandas as pd
from nhanes_dl import download, types
from typing import List
import pandas as pd
from toolz import curry
from nhanes_cvr.transformers import iqrBinaryClassesRemoval
from nhanes_cvr.types import DF, CVSearch, CVTrainDF, GridSearchDF, XYPair


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
    NOT_DEAD = 11


T = TypeVar('T')


def exists(xs: List[T], x: T) -> bool:
    return any([x == y for y in xs])


def toCauseOfDeath(x):
    if np.isnan(x) or x == None:
        return LeadingCauseOfDeath.NOT_DEAD
    return LeadingCauseOfDeath(int(x))


def nhanesToCardiovascularConditions(nhanes_dataset: DF) -> XYPair:
    X = nhanes_dataset.drop(columns=["MCQ160F", "MCQ160C", "MCQ160B", "MCQ160E"]) \
        .drop(columns=download.getMortalityColumns()) \
        .select_dtypes(exclude=['object'])

    hadStroke = nhanes_dataset.MCQ160F == 1  # Stroke
    hadCoronary = nhanes_dataset.MCQ160C == 1  # Coronary Heart Disease
    hadCongestiveHeartFailure = nhanes_dataset.MCQ160B == 1  # Congestive Heart Failure
    hadHeartAttack = nhanes_dataset.MCQ160E == 1  # Heart Attack
    Y = hadStroke | hadCoronary | hadCongestiveHeartFailure | hadHeartAttack

    return (X, Y)


def nhanesHeartFailure(nhanes_dataset: DF) -> pd.Series:
    hadStroke = nhanes_dataset.MCQ160F == 1  # Stroke
    hadCoronary = nhanes_dataset.MCQ160C == 1  # Coronary Heart Disease
    hadCongestiveHeartFailure = nhanes_dataset.MCQ160B == 1  # Congestive Heart Failure
    hadHeartAttack = nhanes_dataset.MCQ160E == 1  # Heart Attack
    return hadStroke | hadCoronary | hadCongestiveHeartFailure | hadHeartAttack


def nhanesCVRDeath(dataset: DF) -> pd.Series:
    # Do not need to check if person has died
    leadingCause = dataset.UCOD_LEADING.apply(toCauseOfDeath)  # type: ignore

    hadHeartDisease = leadingCause == LeadingCauseOfDeath.HEART_DISEASE
    hadCerebrovascular = leadingCause == LeadingCauseOfDeath.CEREBROVASCULAR_DISEASE

    return hadHeartDisease | hadCerebrovascular


def nhanesMortalityContributedDeath(dataset: DF) -> pd.Series:
    hyperTen = dataset.HYPERTEN.fillna(0)
    return hyperTen == 1


def nhanesToMortalitySet(nhanes_dataset: DF) -> XYPair:
    dataset = nhanes_dataset.loc[nhanes_dataset.ELIGSTAT == 1, :]
    dataset = dataset.loc[dataset.MORTSTAT == 1, :]
    leadingCause = dataset.UCOD_LEADING.apply(toCauseOfDeath)  # type: ignore

    hadHeartDisease = leadingCause == LeadingCauseOfDeath.HEART_DISEASE
    hadCerebrovascular = leadingCause == LeadingCauseOfDeath.CEREBROVASCULAR_DISEASE

    Y = hadHeartDisease | hadCerebrovascular
    X = dataset.drop(columns=download.getMortalityColumns()) \
        .select_dtypes(exclude=['object'])

    return (X, Y)


def nhanesToCardiovascularSymptoms(nhanes_dataset: DF) -> XYPair:
    dataset = nhanes_dataset.drop(columns=download.getMortalityColumns())
    discomfortInChest = (dataset.CDQ001 == 1)
    notReliefByStanding = (dataset.CDQ005 == 2)
    painInLeftArm = (dataset.CDQ009G == 7)
    severePainInChest = (dataset.CDQ009H == 9)
    Y = (discomfortInChest | notReliefByStanding |
         painInLeftArm | severePainInChest).astype(int)
    toDrop = ["CDQ001", "CDQ002", "CDQ003",
              "CDQ004", "CDQ005", "CDQ006", "CDQ009A",
              "CDQ009B", "CDQ009C", "CDQ009D", "CDQ009E", "CDQ009F",
              "CDQ009G", "CDQ009H", "CDQ008", "CDQ010"]
    X = dataset.drop(columns=toDrop).select_dtypes(exclude=['object'])

    return (X, Y)


@curry
def nhanesToLabSet(threshold: int, nhanes_dataset: DF) -> XYPair:
    dataset = nhanes_dataset.drop(
        columns=download.getMortalityColumns())
    highChol = (dataset.LBXTC > 239)
    highLDL = (dataset.LBDLDL > 130)
    lowHDL = (dataset.LBDHDD < 40)
    highTG = (dataset.LBXTR > 200)
    highGlucose = (dataset.LBXGLU > 200)
    cvRisk = pd.concat(
        [highChol, highLDL, lowHDL, highGlucose, highTG], axis=1).astype(int).sum(axis=1)
    Y = (cvRisk > threshold).astype(int)
    toDrop = ["LBXTC", "LBDTCSI",
              "LBDLDL", "LBDLDLSI",
              "LBDHDD", "LBDHDDSI",
              "LBXGLU", "LBDGLUSI",
              "LBXTR", "LBDTRSI",
              "LBXSCH", "LBDSCHSI",
              "LBXSGL", "LBDSGLSI",
              "LBXSTR", "LBDSTRSI"]
    X = dataset.drop(columns=toDrop)
    return (X, Y)


def nhanesToHypertensionPaperSet(nhanes_dataset: DF) -> XYPair:
    nhanes_dataset = filterNhanesDatasetByReleaseYears(
        [9, 10], nhanes_dataset)
    hypertenThreshold = 130
    cols = ["RIAGENDR", "RIDAGEYR", "RIDRETH1",
            "BMXBMI", "DIQ010", "SMQ020", "KIQ022"]
    systolicCols = ["BPXSY1", "BPXSY2", "BPXSY3"]

    # Dropping null row
    toDrop = nhanes_dataset.loc[:, systolicCols +
                                ["BMXBMI"]].isna().any(axis=1)
    nhanes_dataset = nhanes_dataset.loc[~toDrop, :]

    # Calc Y
    meanSys = nhanes_dataset.loc[:, systolicCols].mean(axis=1)
    y = meanSys >= hypertenThreshold

    X = nhanes_dataset.loc[:, cols]

    return iqrBinaryClassesRemoval(X, y)


def nhanesToHypertensionContribDeathSet(nhanes_dataset: DF) -> XYPair:
    cols = ["RIAGENDR", "RIDAGEYR", "RIDRETH1",
            "BMXBMI", "DIQ010", "SMQ020", "KIQ022"]
    dataset = nhanes_dataset.loc[nhanes_dataset.ELIGSTAT == 1, :]
    dataset = dataset.loc[dataset.MORTSTAT == 1, :]
    Y = dataset.HYPERTEN.fillna(0)
    X = dataset.loc[:, cols]

    return X, Y


def filterNhanesDatasetByReleaseYears(nhanes_years: List[int], nhanes_dataset: DF) -> DF:
    years = nhanes_dataset.loc[:, "SDDSRVYR"]
    keep = years.apply(lambda x: any([x == y for y in nhanes_years]))

    return nhanes_dataset.loc[keep, :]


def makeDirectoryIfNotExists(directory: str) -> bool:
    import os

    try:
        os.mkdir(directory)
        return True

    except:
        return False


def get_nhanes_dataset() -> DF:
    # cacheDir = "../nhanse-dl/nhanes_cache"
    cacheDir = "./nhanes_cache"
    years = types.allContinuousNHANES()
    NHANES_DATASET = download.readCacheNhanesYearsWithMortality(
        cacheDir, years)

    # Process NHANES
    LINKED_DATASET = NHANES_DATASET.loc[NHANES_DATASET.ELIGSTAT == 1, :]
    DEAD_DATASET = LINKED_DATASET.loc[LINKED_DATASET.MORTSTAT == 1, :]
    ALIVE_DATASET = LINKED_DATASET.loc[LINKED_DATASET.MORTSTAT == 0, :]

    print(f"Entire Dataset: {NHANES_DATASET.shape}")
    print(f"Linked Mortality Dataset: {LINKED_DATASET.shape}")
    print(f"Dead Dataset: {DEAD_DATASET.shape}")
    print(f"Alive Dataset: {ALIVE_DATASET.shape}")
    DEAD_DATASET.describe().to_csv("./results/dead_dataset_info.csv")

    dataset = NHANES_DATASET
    above20AndNonPregnant = (dataset["RIDAGEYR"] >= 20) & (
        dataset["RHD143"] != 1)
    dataset = dataset.loc[above20AndNonPregnant, :]
    dataset = dataset.reset_index(drop=True)
    dataset.describe().to_csv('./results/main_dataset_info.csv')

    print(f"Main Dataset: {dataset.shape}")
    return dataset


def buildDataFrameOfResult(cvSearch: Tuple[str, CVSearch]) -> GridSearchDF:
    n, cv = cvSearch
    res = pd.DataFrame(cv.cv_results_).assign(pipeline=n)
    return GridSearchDF(res)


def buildDataFrameOfResults(cvSearches: List[Tuple[str, CVSearch]]) -> GridSearchDF:
    res = [buildDataFrameOfResult(cv) for cv in cvSearches]
    df = pd.concat(res, ignore_index=True)
    return GridSearchDF(df)


def concatString(xs: List[str]) -> str:
    return reduce(lambda acc, curr: acc + curr, xs, "")
