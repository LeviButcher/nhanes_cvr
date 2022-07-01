from math import ceil, sqrt
import pandas as pd
from sklearn import linear_model, model_selection, preprocessing, pipeline, ensemble, metrics, datasets
from typing import Union, List, NewType, Dict, Any, Tuple, Callable
import matplotlib.pyplot as plt
import seaborn as sns

from nhanes_cvr import utils

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
XSet = pd.DataFrame
YSet = pd.Series
XYPair = Tuple[XSet, YSet]
# Function that Labels Dataset
Labeller = Callable[[pd.DataFrame], XYPair]
# Function that selects features/samples to use for training
Selector = Callable[[XYPair], XYPair]
NamedSelector = Tuple[str, Selector]
NamedSelectors = List[NamedSelector]
NamedLabeller = Tuple[str, Labeller]
SelectorCVTestDF = pd.DataFrame


def generatePipelines(models: List[Tuple[ModelConst, ModelConf]], scaling: List[ScalingConst]) -> CVModelList:
    return [(pipeline.Pipeline([('scaling', s()), ('model', m())]), c)
            for m, c in models for s in scaling]


def getClassName(x):
    return x.__class__.__name__


def randomSearchCV(model: Model, config: ModelConf, scoring: Scoring, refit, cv, X: pd.DataFrame, Y: pd.Series) -> CVSearch:
    clf = model_selection.RandomizedSearchCV(
        estimator=model, param_distributions=config, n_iter=20, scoring=scoring,
        n_jobs=15, refit=refit, cv=cv, return_train_score=True)
    clf.fit(X, Y)
    return clf


def buildDataFrameOfResults(results: List[CVSearch]) -> CVTrainDF:
    res = [buildDataFrameOfResult(res) for res in results]
    return CVTrainDF(pd.concat(res, ignore_index=True))


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
        uniqueConfs = pd.Series(confs).drop_duplicates() \
            .apply(lambda x: f"param_{x}")

        g = sns.pairplot(res, x_vars=uniqueConfs, y_vars=scoring,
                         hue="scaling", kind="scatter")

        def customLinePlot(xdata, ydata, **kwargs):
            sns.lineplot(x=xdata, y=ydata, **kwargs)

        g.map(customLinePlot)

        g.fig.subplots_adjust(top=.9)
        g.fig.suptitle(name)
        g.fig.subplots_adjust(hspace=0.9, wspace=0.9)

        plt.savefig(f"{savePath}_{name}")
        plt.close()


def plotValCurveForModels(df: CVTrainDF, foldCount, savePath):
    idxFoldNumber = 5
    accTrain = [f"split{i}_train_accuracy" for i in range(foldCount)]
    accTest = [f"split{i}_test_accuracy" for i in range(foldCount)]

    bestIdx = [res.rank_test_f1.idxmin()
               for _, res in df.groupby(by=['model', 'scaling'])]
    bestModels = df.loc[bestIdx, :]

    data = []
    for _, res in bestModels.groupby(by=['model', 'scaling']):
        trainScore = res.melt(
            id_vars=['model', 'scaling'], value_vars=accTrain)
        # Index 5 is position of fold number
        trainScore = trainScore.assign(idx=trainScore.variable.apply(
            lambda x: int(x[idxFoldNumber]))).assign(type="train")
        testScore = res.melt(id_vars=['model', 'scaling'], value_vars=accTest)
        testScore = testScore.assign(idx=testScore.variable.apply(
            lambda x: int(x[idxFoldNumber]))).assign(type="test")
        data.append(trainScore)
        data.append(testScore)

    data = pd.concat(data, ignore_index=True)
    g = sns.relplot(data=data, row='model', col='scaling', hue='type',
                    x='idx', y='value', kind='line')
    g.set(xlabel='fold', ylabel='accuracy')
    plt.savefig(savePath)
    plt.close()


def plotTestResults(results: CVTestDF, scoring, title, savePath):
    g = sns.PairGrid(results, y_vars=scoring, x_vars=['model'], hue="scaling")
    g.map(sns.stripplot)
    g.add_legend()

    g.fig.subplots_adjust(top=.9)
    g.fig.suptitle(title)
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


# Remove later if I don't like the new confusion matrix
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


def plotNewConfusionMatrix(results: CVTestDF, title: str, savePath: str):
    g = sns.FacetGrid(results, row="modelAppr", col="scalingAppr")

    # hackish way of getting heatmap drawn
    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop('data')
        sns.heatmap(data.cm[0], **kwargs)

    g.map_dataframe(draw_heatmap, data="cm",
                    annot=True, square=True, cbar=False, cmap="Blues")
    g.fig.subplots_adjust(top=0.9)  # adjust the Figure in rp
    g.fig.suptitle(title)
    g.fig.tight_layout()

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
                    bbox_to_anchor=(1.3, 1))
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
                    bbox_to_anchor=(1.4, 1))
    plt.savefig(savePath, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def evaluateBestModel(model: CVSearch, scoring: Scoring, X, Y) -> CVTestDF:
    scores = [s(model, X, Y) for (_, s) in scoring.items()]
    scoreNames = scoring.keys()
    (modelName, scalingName) = getModelScalerNames(model)
    (mAppr, sAppr) = getModelScalerNamesAppr(model)
    bestParams = model.best_params_
    predictedY = model.predict(X)
    cm = metrics.confusion_matrix(Y, predictedY, normalize='true')
    record = [modelName, scalingName, mAppr, sAppr, *scores, bestParams, cm]
    cols = ['model', 'scaling', 'modelAppr',
            'scalingAppr', *scoreNames, 'params', 'cm']

    return CVTestDF(pd.DataFrame([record], columns=cols))


def evaluateBestModels(models: List[CVSearch], scoring: Scoring, X, Y) -> CVTestDF:
    res = pd.concat([evaluateBestModel(m, scoring, X, Y) for m in models])
    return CVTestDF(res)


def trainTestProcess(cvModels: CVModelList, scoring: Scoring, target: str, testSize: float, cv: Fold, X, Y, saveDir: str) -> CVTestDF:
    utils.makeDirectoryIfNotExists(saveDir)
    scoringNames = scoring.keys()
    cvTrainScores = [f"mean_train_{s}" for s in scoringNames]
    cvTestScores = [f"mean_test_{s}" for s in scoringNames]
    trainX, testX, trainY, testY = model_selection.train_test_split(
        X, Y, test_size=testSize, random_state=42, stratify=Y)

    trainedModels = [randomSearchCV(m, c, scoring, target, cv, trainX, trainY)
                     for m, c in cvModels]
    trainResults = buildDataFrameOfResults(trainedModels)

    trainResults.to_csv(f"{saveDir}train_results.csv")

    plotValCurveForModels(trainResults, 10, f"{saveDir}train_val_curves")
    # Training Plots
    plotPrecisionRecallForModels(
        trainedModels, trainX, trainY, f"{saveDir}train_precision_recall")
    plotROCCurveForModels(trainedModels, trainX, trainY,
                          f"{saveDir}train_roc_curve")
    pairPlotsForModelConfigs(trainResults, cvTrainScores,
                             f"{saveDir}train_train_fold_scores")
    pairPlotsForModelConfigs(trainResults, cvTestScores,
                             f"{saveDir}train_test_fold_scores")

    testResults = evaluateBestModels(trainedModels, scoring, testX, testY)

    testResults.to_csv(f"{saveDir}test_results.csv")
    # Test Plots
    plotTestResults(testResults, scoringNames,
                    "test scores", f"{saveDir}test_scores")
    plotPrecisionRecallForModels(
        trainedModels, testX, testY, f"{saveDir}test_precision_recall")
    plotROCCurveForModels(trainedModels, testX, testY,
                          f"{saveDir}test_roc_curve")

    plotNewConfusionMatrix(testResults, "test - CM",
                           f"{saveDir}test_confusion_matrix")
    return testResults


def test():
    X, Y = datasets.make_classification()

    models = [
        (linear_model.LogisticRegression,
         {
             'model__C': [.5, 1],
             'model__solver': ['lbfgs', 'liblinear']
         }
         ),
        (ensemble.RandomForestClassifier,
         {
             'model__n_estimators': [100, 50],
             'model__criterion': ['gini', 'entropy']
         }
         )]

    scalers = [
        preprocessing.MinMaxScaler,
        preprocessing.Normalizer,
        preprocessing.StandardScaler,
        preprocessing.RobustScaler
    ]

    scoringConfig = {"precision": metrics.make_scorer(metrics.precision_score, average="binary", zero_division=0),
                     "recall": metrics.make_scorer(metrics.recall_score, average="binary", zero_division=0),
                     "f1": metrics.make_scorer(metrics.f1_score, average="binary", zero_division=0),
                     "accuracy": metrics.make_scorer(metrics.accuracy_score)
                     }

    pipes = generatePipelines(models, scalers)
    fold = model_selection.StratifiedKFold(n_splits=10)
    trainTestProcess(pipes, scoringConfig, "f1",
                     0.2, fold, X, Y, "../results")


def plotFeatureRelationships(X, Y, savePath):
    data = X.assign(Y=Y)
    sns.pairplot(data, diag_kind='hist', hue='Y', corner=True)
    plt.savefig(savePath)
    plt.close()


def trainTestWithSelector(namedSelector: NamedSelector, xy: XYPair, models: CVModelList, scoring: Scoring, target: str, testSize: float, cv: Fold, saveDir: str) -> SelectorCVTestDF:
    name, selector = namedSelector
    (X, Y) = selector(xy)
    utils.makeDirectoryIfNotExists(saveDir)
    testResult = trainTestProcess(
        models, scoring, target, testSize, cv, X, Y, f"{saveDir}/{name}/")
    return testResult.assign(selector=name)


def labelThenTrainUsingMultipleSelectors(namedLabeller: NamedLabeller, data: pd.DataFrame, selectors: NamedSelectors, models: CVModelList, scoring: Scoring, target: str, testSize: float, cv: Fold, saveDir: str):
    name, labeller = namedLabeller
    xy = labeller(data)
    print(xy[0].shape)
    print(xy[1].value_counts(normalize=True))
    utils.makeDirectoryIfNotExists(saveDir)

    results = [trainTestWithSelector(ns, xy, models, scoring,
                                     target, testSize, cv, f"{saveDir}/{name}/")
               for ns in selectors]
    results = pd.concat(results).assign(labeller=name)
