from more_itertools import unzip
import pandas as pd
from sklearn import model_selection, metrics
from typing import List
from nhanes_cvr import utils
from imblearn import pipeline
from nhanes_cvr.transformers import DropTransformer
from nhanes_cvr.types import *
from nhanes_cvr import plots


N_JOBS = 5


def generatePipelines(models: List[GenModelConf],
                      scaling: ConstList[Scaling],
                      replacements: ConstList[Replacement],
                      selections: ConstList[Selection]):
    return [(pipeline.Pipeline([
        ('drop', DropTransformer(threshold=0.5)),
        ('replacement', r()),
        ('scaling', s()),
        ('selection', sel()),
        ('model', m())]), c)
        for m, c in models
        for s in scaling
        for r in replacements
        for sel in selections]


# --- TRAINING ---

def randomSearchCV(model: Model, config: ModelConf, scoring: Scoring,
                   refit: str, cv: Folding, X: pd.DataFrame, Y: pd.Series) -> CVSearch:
    clf = model_selection.GridSearchCV(
        estimator=model, param_grid=config, scoring=scoring,
        n_jobs=N_JOBS, refit=refit, cv=cv, return_train_score=True)
    return clf.fit(X, Y)


def evaluateBestModel(model: CVSearch, scoring: Scoring, X: pd.DataFrame, Y: pd.Series) -> CVTestDF:
    scores = [s(model, X, Y) for (_, s) in scoring.items()]
    scoreNames = scoring.keys()
    classes = utils.getPipelineClasses(model)
    classesAppr = utils.getPipelineClassesAppr(model)
    bestParams = model.best_params_
    predictedY = model.predict(X)
    cm = metrics.confusion_matrix(Y, predictedY)
    record = [*classes, *classesAppr, *scores, bestParams, cm]
    cols = ['model', 'scaling', 'selection', 'replacement', 'sampling',
            'modelAppr', 'scalingAppr', 'selectionAppr', 'replacementAppr', 'samplingAppr',
            *scoreNames, 'params', 'cm']

    return CVTestDF(pd.DataFrame([record], columns=cols))


def evaluateBestModels(models: List[CVSearch], scoring: Scoring, X: pd.DataFrame, Y: pd.Series) -> CVTestDF:
    allTestRes = [evaluateBestModel(m, scoring, X, Y) for m in models]
    df = pd.concat(allTestRes, ignore_index=True)
    return CVTestDF(df)


def trainTestProcess(cvModels: CVModelList, scoring: Scoring, target: str,
                     cv: Folding, trainX: pd.DataFrame, trainY: pd.Series,
                     testX: pd.DataFrame, testY: pd.Series, saveDir: str) -> CVRes:
    utils.makeDirectoryIfNotExists(saveDir)
    saveDir = f"{saveDir}/"

    trainedModels = [randomSearchCV(
        m, c, scoring, target, cv, trainX, trainY) for m, c in cvModels]

    trainResults = utils.buildDataFrameOfResults(trainedModels)
    testResults = evaluateBestModels(trainedModels, scoring, testX, testY)

    plots.runAllPlotting(trainResults, testResults, trainedModels,
                         trainX, trainY, testX, testY, scoring, saveDir)

    return (trainResults, testResults)


def runLabeller(namedLabeller: NamedLabeller, models: CVModelList, scoring: Scoring,
                target: str, testSize: float, cv: Fold, data: pd.DataFrame, saveDir: str) -> LabellerRes:
    name, labeller = namedLabeller
    (X, Y) = labeller(data)

    utils.makeDirectoryIfNotExists(saveDir)
    saveDir = f"{saveDir}/{name}"
    utils.makeDirectoryIfNotExists(saveDir)

    trainX, testX, trainY, testY = model_selection.train_test_split(
        X, Y, test_size=testSize, random_state=42, stratify=Y)

    (trainRes, testRes) = trainTestProcess(
        models, scoring, target, cv, trainX, trainY, testX, testY, saveDir)

    return (trainRes.assign(labeller=name), testRes.assign(labeller=name))


def runAllLabellers(labelMethods: NamedLabellerList, pipelines: CVModelList,
                    scoringConfig: Scoring, target: str,
                    testSize: float, fold: Fold, dataset: pd.DataFrame, saveDir: str) -> LabellerRes:
    allResults = [runLabeller(nl, pipelines, scoringConfig, target, testSize, fold, dataset, saveDir)
                  for nl in labelMethods]

    trainRes, testRes = unzip(allResults)

    allTrain = pd.concat(trainRes, ignore_index=True)
    allTest = pd.concat(testRes, ignore_index=True)

    return (allTrain, allTest)
