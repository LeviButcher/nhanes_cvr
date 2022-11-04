from more_itertools import unzip
import pandas as pd
from sklearn import model_selection, metrics
from typing import List
from nhanes_cvr.config import randomState

from sklearn import clone
from nhanes_cvr import utils
from nhanes_cvr.types import *
from nhanes_cvr import plots


N_JOBS = 5
TOP_MODELS_TO_USE = 5


def randomSearchCV(pipeline: PipeLine, config: PipeLinConf, scoring: Scoring,
                   cv: Folding, X: DF, Y: pd.Series) -> CVSearch:
    clf = model_selection.GridSearchCV(
        estimator=pipeline, param_grid=config, scoring=scoring,
        n_jobs=N_JOBS, cv=cv, return_train_score=False, refit=False)
    return clf.fit(X, Y)


def trainForTopFittingParams(pipeline: PipeLine, cvSearch: CVSearch, target: str,
                             modelsToUse: int, X: DF, Y: pd.Series) -> List[PipeLine]:
    def getNewModel(): return clone(pipeline)
    cvResults = DF(cvSearch.cv_results_)
    paramsToUse = cvResults.sort_values(by=f"mean_test_{target}", ascending=False)\
        .head(modelsToUse)\
        .loc[:, "params"]

    return [getNewModel().set_params(**pc).fit(X, Y)  # type: ignore
            for pc in paramsToUse]


def evaluateModel(pipeline: PipeLine, scoring: Scoring, X: DF, Y: pd.Series) -> CVTestDF:
    scores = [s(pipeline, X, Y) for (_, s) in scoring.items()]
    scoreNames = scoring.keys()

    # Assumes CVSearch was trained with a Pipeline
    steps = pipeline.named_steps
    pipelineSteps = steps.keys()
    pipelineValues = steps.values()

    params = pipeline.get_params()
    predictedY = pipeline.predict(X)
    tn, fp, fn, tp = metrics.confusion_matrix(Y, predictedY).ravel()

    cols = [*pipelineSteps, *scoreNames, 'tn', 'fp', 'fn', 'tp', 'params']
    record = [*pipelineValues, *scores, tn, fp, fn, tp, params]

    return CVTestDF(DF([record], columns=cols))


def evaluateAllModels(models: List[PipeLine], scoring: Scoring, X: DF, Y: pd.Series) -> CVTestDF:
    allTestRes = [evaluateModel(m, scoring, X, Y) for m in models]
    df = pd.concat(allTestRes, ignore_index=True)
    return CVTestDF(df)


def trainTestProcess(cvModels: List[PipeLineCV], scoring: Scoring, target: str,
                     cv: Folding, trainX: DF, trainY: pd.Series,
                     testX: DF, testY: pd.Series, saveDir: str) -> CVRes:
    utils.makeDirectoryIfNotExists(saveDir)
    saveDir = f"{saveDir}/"

    cvSearchesWithPipeline = [(m, randomSearchCV(m, c, scoring, cv, trainX, trainY))
                              for m, c in cvModels]
    cvSearches = [c for _, c in cvSearchesWithPipeline]

    trainedModels = [p
                     for pipeline, cvs in cvSearchesWithPipeline
                     for p in trainForTopFittingParams(pipeline, cvs, target, TOP_MODELS_TO_USE, trainX, trainY)]

    trainResults = utils.buildDataFrameOfResults(cvSearches)
    testResults = evaluateAllModels(trainedModels, scoring, testX, testY)

    plots.runAllPlotting(trainResults, testResults, cvSearches,
                         trainX, trainY, testX, testY, scoring, saveDir)

    return (trainResults, testResults)


def runLabeller(namedLabeller: NamedLabeller, pipelines: List[PipeLineCV], scoring: Scoring,
                target: str, testSize: float, cv: Fold, data: DF, saveDir: str) -> LabellerRes:
    name, labeller = namedLabeller
    (X, Y) = labeller(data)

    utils.makeDirectoryIfNotExists(saveDir)
    saveDir = f"{saveDir}/{name}"
    utils.makeDirectoryIfNotExists(saveDir)

    trainX, testX, trainY, testY = model_selection.train_test_split(
        X, Y, test_size=testSize, random_state=randomState, stratify=Y)

    trainX = DF(trainX)
    testX = DF(testX)
    trainY = pd.Series(trainY)
    testY = pd.Series(testY)

    (trainRes, testRes) = trainTestProcess(
        pipelines, scoring, target, cv,
        trainX, trainY, testX, testY, saveDir)
    trainRes = LabellerTrainDF(trainRes.assign(labeller=name))
    testRes = LabellerTestDF(testRes.assign(labeller=name))

    return (trainRes, testRes)


def runAllLabellers(labelMethods: NamedLabellerList, pipelines: List[PipeLineCV],
                    scoringConfig: Scoring, target: str,
                    testSize: float, fold: Fold, dataset: DF, saveDir: str) -> LabellerRes:
    allResults = [runLabeller(nl, pipelines, scoringConfig, target, testSize, fold, dataset, saveDir)
                  for nl in labelMethods]

    trainRes, testRes = unzip(allResults)

    allTrain = LabellerTrainDF(pd.concat(trainRes, ignore_index=True))
    allTest = LabellerTestDF(pd.concat(testRes, ignore_index=True))

    return (allTrain, allTest)
