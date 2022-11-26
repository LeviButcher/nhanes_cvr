from matplotlib import patches
import pandas as pd
from sklearn import model_selection, metrics
from typing import List
from sklearn import clone
from nhanes_cvr import utils
from nhanes_cvr.types import *
from nhanes_cvr import plots
import matplotlib.pyplot as plt
import copy
import seaborn as sns


N_JOBS = 5
TOP_MODELS_TO_USE = 5
randomState = 42


def randomSearchCV(pipeline: PipeLine, config: PipeLinConf, scoring: Scoring,
                   cv: Folding, X: DF, Y: pd.Series) -> CVSearch:
    # Necessary to ensure no reused references between label runs
    pipeline = clone(pipeline)  # type: ignore
    config = copy.deepcopy(config)
    clf = model_selection.GridSearchCV(
        estimator=pipeline, param_grid=config, scoring=scoring,
        n_jobs=N_JOBS, cv=cv, return_train_score=False, refit=False)
    return clf.fit(X, Y)


def trainForTopFittingParams(cvSearch: CVSearch, target: str,
                             modelsToUse: int, X: DF, Y: pd.Series) -> List[PipeLine]:
    def getNewModel(): return clone(cvSearch.estimator)
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


"""
Return back best performing pipeline
"""


def trainTestProcess(cvModels: List[PipeLineCV], scoring: Scoring, target: str,
                     cv: Folding, trainX: DF, trainY: pd.Series,
                     testX: DF, testY: pd.Series, saveDir: str) -> PipeLine:
    utils.makeDirectoryIfNotExists(saveDir)
    saveDir = f"{saveDir}/"

    cvSearches = [randomSearchCV(m, c, scoring, cv, trainX, trainY)
                  for m, c in cvModels]

    trainedModels = [p
                     for cvs in cvSearches
                     for p in trainForTopFittingParams(cvs, target, TOP_MODELS_TO_USE, trainX, trainY)]

    trainResults = utils.buildDataFrameOfResults(cvSearches)
    testResults = evaluateAllModels(trainedModels, scoring, testX, testY)

    # May need checked
    bestIdx: int = testResults['f1'].idxmax()  # type: ignore
    bestTestModel = trainedModels[bestIdx]

    plots.runAllPlotting(trainResults, testResults, trainedModels,
                         trainX, trainY, testX, testY, scoring, saveDir)

    # bestFeatures = pd.Series(
    #     bestTestModel[:-1].n_features_in_)
    # bestFeatures.to_csv(f"{saveDir}chosen_features.csv")

    return bestTestModel


def runLabeller(namedLabeller: NamedLabeller, pipelines: List[PipeLineCV], scoring: Scoring,
                target: str, testSize: float, cv: Fold, dataset: DF, saveDir: str) -> PipeLine:
    name, labeller = namedLabeller
    (X, Y) = labeller(dataset)

    utils.makeDirectoryIfNotExists(saveDir)
    saveDir = f"{saveDir}/{name}"
    utils.makeDirectoryIfNotExists(saveDir)

    trainX, testX, trainY, testY = model_selection.train_test_split(
        X, Y, test_size=testSize, random_state=randomState, stratify=Y)

    trainX = DF(trainX)
    testX = DF(testX)
    trainY = pd.Series(trainY)
    testY = pd.Series(testY)
    print(trainX.shape)
    print(testX.shape)

    bestModel = trainTestProcess(
        pipelines, scoring, target, cv,
        trainX, trainY, testX, testY, saveDir)

    return bestModel


def runRiskAnalyses(name: str, labelMethods: NamedLabellerList, pipelines: List[PipeLineCV],
                    scoringConfig: Scoring, target: str,
                    testSize: float, fold: Fold, dataset: DF, riskFunction: GetRisk, saveDir: str):
    # May need to stratify here
    coreDataset, riskDataset = model_selection.train_test_split(
        dataset, test_size=.1, random_state=randomState)
    coreDataset = pd.DataFrame(coreDataset)
    riskDataset = pd.DataFrame(riskDataset)

    bestModelsWithLabeller = [(nl, runLabeller(nl, pipelines, scoringConfig, target, testSize, fold, coreDataset, saveDir))
                              for nl in labelMethods]

    riskPredictions = []
    for ((_, labeller), model) in bestModelsWithLabeller:
        (xCore, _) = labeller(coreDataset)
        riskX = riskDataset.loc[:, xCore.columns.to_list()]
        res = model.predict(riskX)
        riskPredictions.append(res)

    riskScore = pd.DataFrame(riskPredictions).sum(axis=0)

    riskLabel = riskFunction(riskDataset)
    df = pd.DataFrame({'score': riskScore.to_list(),
                      'risk': riskLabel.to_list()})

    totalScore = df.score.value_counts()
    totalRisk = df.groupby('score').sum().reset_index()
    totalRisk = totalRisk.assign(total=totalScore)

    sns.barplot(data=totalRisk, x="score", y="total", color="lightblue")
    sns.barplot(data=totalRisk, x="score", y="risk", color="darkblue")
    plt.ylabel('Count')
    top_bar = patches.Patch(color='darkblue', label='Risk = Yes')
    bottom_bar = patches.Patch(color='lightblue', label='Risk = No')
    plt.legend(handles=[top_bar, bottom_bar])
    plt.title(name)

    totalRisk.to_csv(f"{saveDir}/riskAnalyses.csv")
    plt.savefig(f"{saveDir}/riskAnalyses_plot.png")
    plt.close()
