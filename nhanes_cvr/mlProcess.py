from math import ceil, sqrt
import pandas as pd
import numpy as np
from sklearn import linear_model, model_selection, preprocessing, datasets, pipeline, ensemble, metrics
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from typing import Union, List, NewType, Dict, Any, Tuple, Callable
import matplotlib.pyplot as plt
import seaborn as sns

# Types
CVSearch = Union[model_selection.GridSearchCV,
                 model_selection.RandomizedSearchCV]
CVTrainDF = NewType('CVTrainDF', pd.DataFrame)
CVTestDF = NewType('CVTestDF', pd.DataFrame)
Scoring = Dict[str, Any]
ModelConf = Dict[str, List[Any]]
Model = Union[linear_model.LogisticRegression,
              ensemble.RandomForestClassifier, pipeline.Pipeline]
Scaling = Union[preprocessing.StandardScaler, preprocessing.MinMaxScaler]
CVModel = Tuple[Model, ModelConf]
CVModelList = List[CVModel]
Fold = model_selection.StratifiedKFold
ModelConst = Callable[[], Model]
ScalingConst = Callable[[], Scaling]


def generatePipelines(models: List[Tuple[ModelConst, ModelConf]], scaling: List[ScalingConst]) -> CVModelList:
    return [(pipeline.Pipeline([('scaling', s()), ('model', m())]), c)
            for m, c in models for s in scaling]


def getClassName(x):
    return x.__class__.__name__


def randomSearchCV(model: Model, config: ModelConf, scoring: Scoring, refit, cv, X: pd.DataFrame, Y: pd.Series) -> CVSearch:
    clf = model_selection.RandomizedSearchCV(
        estimator=model, param_distributions=config, n_iter=50, scoring=scoring,
        n_jobs=-1, refit=refit, cv=cv, return_train_score=True)
    clf.fit(X, Y)
    return clf


def buildDataFrameOfResults(results: List[CVSearch]) -> CVTrainDF:
    res = [buildDataFrameOfResult(res) for res in results]
    return CVTrainDF(pd.concat(res))


def buildDataFrameOfResult(res: CVSearch) -> CVTrainDF:
    return CVTrainDF(pd.DataFrame(res.cv_results_)
                     .assign(model=getClassName(res.estimator['model']),
                             scaling=getClassName(res.estimator['scaling'])))


def pairPlotsForModelConfigs(df: CVTrainDF, scoring: List[str], savePath) -> None:
    # May have to change to accommodate model config
    # TODO Make this into line plots
    for name, res in df.groupby(by="model"):
        confs = res.params.apply(lambda x: list(x.keys())).sum()
        confs = [x for x in confs if x != "model__random_state"]
        uniqueConfs = pd.Series(confs).drop_duplicates().apply(
            lambda x: f"param_{x}")

        g = sns.pairplot(res, x_vars=uniqueConfs, y_vars=scoring,
                         hue="scaling", kind="scatter")

        # move overall title up
        g.fig.subplots_adjust(top=.9)

        # add overall title
        g.fig.suptitle(name)
        plt.savefig(f"{savePath}_{name}")
        plt.close()


def plotTestResults(results: CVTestDF, scoring, savePath):
    res = results.melt(id_vars=['model', 'scaling'], value_vars=scoring)
    g = sns.relplot(data=res, x="model", y='value',
                    col='variable', hue='scaling', kind="scatter")
    g.fig.suptitle("Test Scores")
    plt.savefig(savePath)
    plt.close()


def onlyUpperCase(xs: str) -> str:
    return "".join([x for x in xs if x.isupper()])


def getModelScalerNames(model: CVSearch) -> Tuple[str, str]:
    m = onlyUpperCase(getClassName(model.estimator['model']))
    s = onlyUpperCase(getClassName(model.estimator['scaling']))
    return (m, s)


def getModelScalerNamesAppr(model: CVSearch) -> Tuple[str, str]:
    (m, s) = getModelScalerNames(model)

    return (onlyUpperCase(m), onlyUpperCase(s))


def plotConfusionMatrixForModels(models: List[CVSearch], X, Y, title, savePath):
    total = len(models)
    rowcol = ceil(sqrt(total))
    fig, axs = plt.subplots(rowcol, rowcol, sharex=True, sharey=True)
    axs = axs.flatten()
    fig.tight_layout(pad=10)
    for i, m in enumerate(models):
        ax = axs[i]  # type: ignore
        modelName, scalerName = getModelScalerNamesAppr(m)
        metrics.ConfusionMatrixDisplay.from_estimator(
            m, X, Y, ax=ax, colorbar=False, normalize='true')
        ax.set_title(f"{modelName} - {scalerName}")

    for ax in axs.flat:
        ax.label_outer()

    plt.title(title)

    plt.savefig(savePath)
    plt.close()


def plotPrecisionRecallForModels(models: List[CVSearch], X, Y, savePath):
    ax = plt.gca()
    for m in models:
        modelName, scalerName = getModelScalerNamesAppr(m)
        metrics.PrecisionRecallDisplay.from_estimator(
            m, X, Y, ax=ax, name=f"{modelName} - {scalerName}")

    ax.set_title("precision recall curve")
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center',
                    bbox_to_anchor=(1.3, 0.6))
    plt.savefig(savePath, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def plotROCCurveForModels(models: List[CVSearch], X, Y, savePath):
    _, ax = plt.subplots(1, sharex=True, sharey=True)
    for m in models:
        modelName, scalerName = getModelScalerNamesAppr(m)
        name = f"{modelName} - {scalerName}"
        metrics.RocCurveDisplay.from_estimator(
            m, X, Y, ax=ax, name=name)

    ax.set_title("roc curve")
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center',
                    bbox_to_anchor=(1.4, 0.6))
    plt.savefig(savePath, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def evaluateBestModel(model: CVSearch, scoring: Scoring, X, Y) -> CVTestDF:
    scores = [s(model, X, Y) for (_, s) in scoring.items()]
    scoreNames = scoring.keys()
    modelName = getClassName(model.estimator['model'])
    scalingName = getClassName(model.estimator['scaling'])
    bestParams = model.best_params_
    record = [modelName, scalingName, *scores, bestParams]
    cols = ['model', 'scaling', *scoreNames, 'params']
    return CVTestDF(pd.DataFrame([record], columns=cols))


def evaluateBestModels(models: List[CVSearch], scoring: Scoring, X, Y) -> CVTestDF:
    res = pd.concat([evaluateBestModel(m, scoring, X, Y) for m in models])
    return CVTestDF(res)


def trainTestProcess(cvModels: CVModelList, scoring: Scoring, target: str, testSize: float, cv: Fold, X, Y, saveDir: str) -> None:
    scoringNames = scoring.keys()
    cvTrainScores = [f"mean_train_{s}" for s in scoringNames]
    cvTestScores = [f"mean_test_{s}" for s in scoringNames]
    trainX, testX, trainY, testY = model_selection.train_test_split(
        X, Y, test_size=testSize)

    trainedModels = [randomSearchCV(m, c, scoring, target, cv, trainX, trainY)
                     for m, c in cvModels]
    trainResults = buildDataFrameOfResults(trainedModels)

    trainResults.to_csv(f"{saveDir}/train_results.csv")
    # Training Plots
    plotConfusionMatrixForModels(
        trainedModels, trainX, trainY, "train - CM", f"{saveDir}/train_confusion_matrix")
    plotPrecisionRecallForModels(
        trainedModels, trainX, trainY, f"{saveDir}/train_precision_recall")
    plotROCCurveForModels(trainedModels, trainX, trainY,
                          f"{saveDir}/train_roc_curve")
    pairPlotsForModelConfigs(trainResults, cvTrainScores,
                             f"{saveDir}/train_train_fold_scores")
    pairPlotsForModelConfigs(trainResults, cvTestScores,
                             f"{saveDir}/train_test_fold_scores")

    testResults = evaluateBestModels(trainedModels, scoring, testX, testY)

    testResults.to_csv(f"{saveDir}/test_results.csv")
    # Test Plots
    plotTestResults(testResults, scoringNames, f"{saveDir}/test_scores")
    plotConfusionMatrixForModels(
        trainedModels, testX, testY, "test - CM", f"{saveDir}/test_confusion_matrix")
    plotPrecisionRecallForModels(
        trainedModels, testX, testY, f"{saveDir}/test_precision_recall")
    plotROCCurveForModels(trainedModels, testX, testY,
                          f"{saveDir}/test_roc_curve")
