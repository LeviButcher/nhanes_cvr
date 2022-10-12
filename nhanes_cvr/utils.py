from enum import Enum
from functools import reduce
from typing import Callable, List, Tuple, TypeVar
from imblearn import FunctionSampler
import numpy as np
import pandas as pd
from scipy import stats
from nhanes_dl import download, types
from typing import List
import pandas as pd
from toolz import curry
import nhanes_cvr.transformers as trans
from nhanes_cvr.transformers import iqrBinaryClassesRemoval
from nhanes_cvr.types import CVSearch, CVTrainDF, XYPair


def getClassName(x) -> str:
    if x is None:
        return "None"

    name = x.__class__.__name__

    if (name == "Pipeline"):
        stepClasses = [getClassName(y) for y in x.named_steps.values()]
        return '_'.join(stepClasses)

    return name


def onlyUpperCase(xs: str) -> str:
    return "".join([x for x in xs if x.isupper()])


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
    NOT_DEAD = 11


GETY = Callable[[pd.DataFrame], int]


@curry
def labelY(func: GETY, dataset: pd.DataFrame) -> pd.Series:
    return dataset.agg(func, axis=1)


T = TypeVar('T')


def exists(xs: List[T], x: T) -> bool:
    return any([x == y for y in xs])


def toCauseOfDeath(x):
    if np.isnan(x) or x == None:
        return LeadingCauseOfDeath.NOT_DEAD
    return LeadingCauseOfDeath(int(x))


def labelQuestionnaireSet(nhanes_dataset) -> Tuple[pd.DataFrame, pd.Series]:
    X = nhanes_dataset.drop(columns=["MCQ160F", "MCQ160C", "MCQ160B", "MCQ160E"]) \
        .drop(columns=download.getMortalityColumns()) \
        .select_dtypes(exclude=['object'])

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


def labelCVrBasedOnNHANESMortalityAndExtraFactors(nhanes_dataset) -> XYPair:
    dataset = nhanes_dataset.loc[nhanes_dataset.ELIGSTAT == 1, :]
    dataset = dataset.loc[dataset.MORTSTAT == 1, :]
    X = dataset.drop(columns=download.getMortalityColumns())
    mortStat = dataset.UCOD_LEADING.apply(toCauseOfDeath)
    cvrDeath = (mortStat == LeadingCauseOfDeath.HEART_DISEASE) \
        | (mortStat == LeadingCauseOfDeath.CEREBROVASCULAR_DISEASE) \
        | (mortStat == LeadingCauseOfDeath.DIABETES_MELLITUS)
    diabetesContrib = dataset.DIABETES == 1
    hypertenContrib = dataset.HYPERTEN == 1
    Y = cvrDeath | diabetesContrib | hypertenContrib

    return (X, Y)


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
    notReliefByStanding = (dataset.CDQ005 == 2)
    painInLeftArm = (dataset.CDQ009G == 7)
    severePainInChest = (dataset.CDQ009H == 9)
    Y = (discomfortInChest | notReliefByStanding |
         painInLeftArm | severePainInChest).astype(int)
    toDrop = ["CDQ001", "CDQ002", "CDQ003",
              "CDQ004", "CDQ005", "CDQ006", "CDQ009A",
              "CDQ009B", "CDQ009C", "CDQ009D", "CDQ009E", "CDQ009F",
              "CDQ009G", "CDQ009H", "CDQ008", "CDQ010"]
    X = dataset.drop(columns=toDrop)
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


def labelCVRViaCVRDeath(dataset) -> pd.Series:
    normalCVRDeath = [LeadingCauseOfDeath.HEART_DISEASE,
                      LeadingCauseOfDeath.CEREBROVASCULAR_DISEASE]
    Y = labelCVR(normalCVRDeath, dataset)  # type: ignore
    return Y  # type: ignore


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


def filterNhanesDatasetByReleaseYears(nhanes_years: List[int], nhanes_dataset: pd.DataFrame) -> pd.DataFrame:
    years = nhanes_dataset.loc[:, "SDDSRVYR"]
    keep = years.apply(lambda x: any([x == y for y in nhanes_years]))

    return nhanes_dataset.loc[keep, :]


def labelHypertensionBasedOnPaper(nhanes_dataset: pd.DataFrame) -> XYPair:
    nhanes_dataset = filterNhanesDatasetByReleaseYears(
        [9, 10], nhanes_dataset)
    hypertenThreshold = 130
    cols = ["RIAGENDR", "RIDAGEYR", "RIDRETH1",
            "BMXBMI", "DIQ010", "SMQ020", "KIQ022"]
    systolicCols = ["BPXSY1", "BPXSY2", "BPXSY3"]

    # Dropping null columns
    toDrop = nhanes_dataset.loc[:, systolicCols +
                                ["BMXBMI"]].isna().any(axis=1)
    nhanes_dataset = nhanes_dataset.loc[~toDrop, :]

    # Calc Y
    meanSys = nhanes_dataset.loc[:, systolicCols].mean(axis=1)
    y = meanSys >= hypertenThreshold

    X = nhanes_dataset.loc[:, cols]

    return iqrBinaryClassesRemoval(X, y)


def removeOutliers(z_score, df, columns=None):
    columns = columns if columns is not None else df.columns
    scores = np.abs(stats.zscore(df.loc[:, columns]))
    return (scores < z_score).all(axis=1)


def makeDirectoryIfNotExists(directory: str) -> bool:
    import os

    try:
        os.mkdir(directory)
        return True

    except:
        return False


def get_nhanes_dataset() -> pd.DataFrame:
    cacheDir = "../nhanse-dl/nhanes_cache"
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


def generateKMeansUnderSampling(kValues, clusterMethods, bestScoresFuncs):
    return [(f"{cm.__name__}_undersampling_{k}_{bestScore.__name__}",
            lambda: FunctionSampler(
                func=trans.kMeansUnderSampling,
                kw_args={'k': k, 'findBest': bestScore, 'clusterMethod': cm}))
            for k in kValues
            for cm in clusterMethods
            for bestScore in bestScoresFuncs]


def pipelineSteps() -> List[str]:
    return ['model', 'scaling', 'selection', 'replacement', 'sampling']


def getPipelineClasses(model: CVSearch) -> List[str]:
    return [getPipelineStepClass(model, s) for s in pipelineSteps()]


def getPipelineClassesAppr(model: CVSearch) -> List[str]:
    return [onlyUpperCase(s) for s in getPipelineClasses(model)]


def getPipelineStepClass(model: CVSearch, step: str):
    return getClassName(model.estimator.named_steps.get(step))


def getPipelineStepClassAppr(model: CVSearch, step: str):
    return onlyUpperCase(getPipelineStepClass(model, step))


def buildDataFrameOfResults(results: List[CVSearch]) -> CVTrainDF:
    res = [buildDataFrameOfResult(res) for res in results]
    return CVTrainDF(pd.concat(res, ignore_index=True))


def buildDataFrameOfResult(res: CVSearch) -> CVTrainDF:
    res = pd.DataFrame(res.cv_results_) \
        .assign(model=getPipelineStepClass(res, 'model'),
                scaling=getPipelineStepClass(res, 'scaling'),
                sampling=getPipelineStepClass(res, 'sampling'),
                selection=getPipelineStepClass(res, 'selection'),
                replacement=getPipelineStepClass(res, 'replacement'))
    return CVTrainDF(res)


def concatString(xs: List[str]) -> str:
    return reduce(lambda acc, curr: acc + curr, xs, "")
