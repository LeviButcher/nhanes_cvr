import functools
import pandas as pd
from sklearn import feature_selection, linear_model, model_selection, preprocessing, ensemble, metrics, impute
from typing import TypeVar, Union, List, NewType, Dict, Any, Tuple, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from nhanes_cvr import utils
from imblearn.under_sampling import RandomUnderSampler
from imblearn import pipeline
from imblearn import FunctionSampler
from nhanes_cvr.transformers import DropTransformer, iqrBinaryClassesRemoval, iqrRemoval
from nhanes_cvr.types import *

# --- UTILS ---


def generatePipelines(models: List[GenModelConf], scaling: ConstList[Scaling],
                      replacements: ConstList[Replacement], selections: ConstList[Selection]):
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


def generatePipelinesWithSampling(models: List[GenModelConf], scaling: ConstList[Scaling],
                                  replacements: ConstList[Replacement], samplings: ConstList[Sampler],
                                  selections: ConstList[Selection]):
    return [(pipeline.Pipeline([
        ('drop', DropTransformer(threshold=0.5)),
        ('replacement', r()),
        ('scaling', s()),
        ('selection', sel()),
        ('sampling', samp()),
        ('model', m())]), c)
        for m, c in models
        for s in scaling
        for r in replacements
        for samp in samplings
        for sel in selections]


def generatePipelinesWithSamplingAndOutlier(models: List[GenModelConf], scaling: ConstList[Scaling],
                                            replacements: ConstList[Replacement], samplings: ConstList[Sampler],
                                            selections: ConstList[Selection], outliers: ConstList[Outlier]):
    return [(pipeline.Pipeline([
        ('drop', DropTransformer(threshold=0.5)),
        ('replacement', r()),
        ('scaling', s()),
        ('selection', sel()),
        ('outlier', out()),
        ('sampling', samp()),
        ('model', m())]), c)
        for m, c in models
        for s in scaling
        for r in replacements
        for samp in samplings
        for sel in selections
        for out in outliers]


def getClassName(x):
    return x.__class__.__name__


def onlyUpperCase(xs: str) -> str:
    return "".join([x for x in xs if x.isupper()])


def pipelineSteps() -> List[str]:
    return ['model', 'scaling', 'selection', 'replacement', 'sampling']


def getPipelineClasses(model: CVSearch) -> List[str]:
    return [getPipelineStepClass(model, s) for s in pipelineSteps()]


def getPipelineClassesAppr(model: CVSearch) -> List[str]:
    return [onlyUpperCase(s) for s in getPipelineClasses(model)]


# --- TRAINING ---


def randomSearchCV(model: Model, config: ModelConf, scoring: Scoring,
                   refit: str, cv: Folding, X: pd.DataFrame, Y: pd.Series) -> CVSearch:
    clf = model_selection.RandomizedSearchCV(
        estimator=model, param_distributions=config, n_iter=20, scoring=scoring,
        n_jobs=5, refit=refit, cv=cv, return_train_score=True)
    return clf.fit(X, Y)


def buildDataFrameOfResults(results: List[CVSearch]) -> CVTrainDF:
    res = [buildDataFrameOfResult(res) for res in results]
    return CVTrainDF(pd.concat(res, ignore_index=True))


def getPipelineStepClass(model: CVSearch, step: str):
    return getClassName(model.estimator.named_steps.get(step))


def getPipelineStepClassAppr(model: CVSearch, step: str):
    return onlyUpperCase(getPipelineStepClass(model, step))


def buildDataFrameOfResult(res: CVSearch) -> CVTrainDF:
    return CVTrainDF(pd.DataFrame(res.cv_results_)
                     .assign(model=getPipelineStepClass(res, 'model'),
                     scaling=getPipelineStepClass(res, 'scaling'),
                     sampling=getPipelineStepClass(res, 'sampling'),
                     selection=getPipelineStepClass(res, 'selection'),
                     replacement=getPipelineStepClass(res, 'replacement')))


def evaluateBestModel(model: CVSearch, scoring: Scoring, X: pd.DataFrame, Y: pd.Series) -> CVTestDF:
    scores = [s(model, X, Y) for (_, s) in scoring.items()]
    scoreNames = scoring.keys()
    classes = getPipelineClasses(model)
    classesAppr = getPipelineClassesAppr(model)
    bestParams = model.best_params_
    predictedY = model.predict(X)
    cm = metrics.confusion_matrix(Y, predictedY)
    record = [*classes, *classesAppr, *scores, bestParams, cm]
    cols = ['model', 'scaling', 'selection', 'replacement', 'sampling',
            'modelAppr', 'scalingAppr', 'selectionAppr', 'replacementAppr', 'samplingAppr',
            *scoreNames, 'params', 'cm']

    return CVTestDF(pd.DataFrame([record], columns=cols))


def evaluateBestModels(models: List[CVSearch], scoring: Scoring, X: pd.DataFrame, Y: pd.Series) -> CVTestDF:
    res = pd.concat([evaluateBestModel(m, scoring, X, Y) for m in models])
    return CVTestDF(res)


def trainTestProcess(cvModels: CVModelList, scoring: Scoring, target: str,
                     cv: Folding, trainX: pd.DataFrame, trainY: pd.Series,
                     testX: pd.DataFrame, testY: pd.Series, saveDir: str) -> CVTestDF:
    utils.makeDirectoryIfNotExists(saveDir)
    scoringNames = list(scoring.keys())
    # cvTrainScores = [f"mean_train_{s}" for s in scoringNames]
    # cvTestScores = [f"mean_test_{s}" for s in scoringNames]
    saveDir = f"{saveDir}/"
    features = trainX.columns.to_list()

    trainedModels = [randomSearchCV(m, c, scoring, target, cv, trainX, trainY)
                     for m, c in cvModels]

    trainResults = buildDataFrameOfResults(trainedModels)

    trainResults.to_csv(f"{saveDir}train_results.csv")

    plotValCurveForModels(trainResults, 10, f"{saveDir}train_val_curves")
    # randomForestFeatureImportance(
    #     trainedModels, features, f"{saveDir}random_forest_importance.csv")
    # Training Plots
    plotPrecisionRecallForModels(
        trainedModels, trainX, trainY, f"{saveDir}train_precision_recall")
    plotROCCurveForModels(trainedModels, trainX, trainY,
                          f"{saveDir}train_roc_curve")
    # pairPlotsForModelConfigs(trainResults, cvTrainScores,
    #                          f"{saveDir}train_train_fold_scores")
    # pairPlotsForModelConfigs(trainResults, cvTestScores,
    #                          f"{saveDir}train_test_fold_scores")

    testResults = evaluateBestModels(trainedModels, scoring, testX, testY)

    testResults.to_csv(f"{saveDir}test_results.csv")
    # Test Plots
    plotTestResults(testResults, scoringNames,
                    f"{saveDir} - test scores", f"{saveDir}test_scores")
    plotPrecisionRecallForModels(
        trainedModels, testX, testY, f"{saveDir}test_precision_recall")
    plotROCCurveForModels(trainedModels, testX, testY,
                          f"{saveDir}test_roc_curve")

    plotConfusionMatrix(testResults, "test - CM",
                        f"{saveDir}test_confusion_matrix")
    return testResults


def randomForestFeatureImportance(models: List[CVSearch], features: List[str], savePath: str):
    rfModels = [m for m in models if getPipelineClasses(m)[
        0] == "RandomForestClassifier"]

    results = []
    for rf in rfModels:
        imp = rf.best_estimator_['model'].feature_importances_  # type: ignore
        data = pd.DataFrame([imp], columns=features) \
            .assign(model='RandomForestClassifier') \
            .assign(scaling=rf.best_estimator_['scaling'])  # type: ignore
        results.append(data)
    results = pd.concat(results)
    results.to_csv(f"{savePath}")


def labelThenTrainTest(namedLabeller: NamedLabeller, models: CVModelList, scoring: Scoring,
                       target: str, testSize: float, cv: Fold, data: pd.DataFrame, saveDir: str):
    name, labeller = namedLabeller
    (X, Y) = labeller(data)

    # TODO: Remove Later - This allows mimicking how the hypertension kMeans paper did their outlier removal
    X, Y = iqrBinaryClassesRemoval(X, Y)

    utils.makeDirectoryIfNotExists(saveDir)
    saveDir = f"{saveDir}/{name}"
    utils.makeDirectoryIfNotExists(saveDir)

    trainX, testX, trainY, testY = model_selection.train_test_split(
        X, Y, test_size=testSize, random_state=42, stratify=Y)

    return trainTestProcess(
        models, scoring, target, cv, trainX, trainY, testX, testY, saveDir)


def train_test_idx(testSize: float, dataset: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    train, test = model_selection.train_test_split(
        dataset.index, test_size=testSize, random_state=42)
    return (train, test)


def concatString(xs: List[str]) -> str:
    return functools.reduce(lambda acc, curr: acc + curr, xs, "")


# --- PLOTTING ---


def pairPlotsForModelConfigs(df: CVTrainDF, scoring: List[str], savePath: str):
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


def plotValCurveForModels(df: CVTrainDF, foldCount: int, savePath: str):
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
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.fig.tight_layout()
    plt.savefig(savePath)
    plt.close()


def plotTestResults(results: CVTestDF, scoring: List[str], title: str, savePath: str):
    fig, axes = plt.subplots(1, len(scoring), figsize=(10, 5))
    for score, ax in zip(scoring, axes):
        sns.scatterplot(data=results, x="modelAppr",
                        y=score, hue="scaling", ax=ax, legend=None)

    axes[1].set_title(title)
    plt.tight_layout()

    plt.savefig(savePath)
    plt.close()


def plotConfusionMatrix(results: CVTestDF, title: str, savePath: str):
    g = sns.FacetGrid(results, row="modelAppr",
                      col="scalingAppr")

    # hackish way of getting heatmap drawn
    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop('data')
        sns.heatmap(data.cm[0], **kwargs)

    g.map_dataframe(draw_heatmap, data="cm",
                    annot=True, fmt="d", square=True, cbar=False, cmap="Blues")
    g.fig.subplots_adjust(top=0.9)  # adjust the Figure in rp
    g.fig.suptitle(title)
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.fig.tight_layout()

    plt.savefig(savePath)
    plt.close()


def plotPrecisionRecallForModels(models: List[CVSearch], X: pd.DataFrame, Y: pd.Series, savePath: str):
    ax = plt.gca()
    for m in models:
        modelName, scalerName, *_ = getPipelineClassesAppr(m)
        metrics.PrecisionRecallDisplay.from_estimator(
            m, X, Y, ax=ax, name=f"{modelName} - {scalerName}")

    ax.set_title("precision recall curve")
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center',
                    bbox_to_anchor=(1.3, 1))
    plt.savefig(savePath, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def plotROCCurveForModels(models: List[CVSearch], X: pd.DataFrame, Y: pd.Series, savePath: str):
    _, ax = plt.subplots(1, sharex=True, sharey=True)
    for m in models:
        modelName, scalerName, *_ = getPipelineClassesAppr(m)
        name = f"{modelName} - {scalerName}"
        metrics.RocCurveDisplay.from_estimator(
            m, X, Y, ax=ax, name=name)

    ax.set_title("roc curve")
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center',
                    bbox_to_anchor=(1.4, 1))
    plt.savefig(savePath, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def plotFeatureRelationships(X: pd.DataFrame, Y: pd.Series, savePath: str):
    data = X.assign(Y=Y)
    sns.pairplot(data, diag_kind='hist', hue='Y', corner=True)
    plt.savefig(savePath)
    plt.close()

# def test():
#     X, Y = datasets.make_classification()

#     models = [
#         (linear_model.LogisticRegression,
#          {
#              'model__C': [.5, 1],
#              'model__solver': ['lbfgs', 'liblinear']
#          }
#          ),
#         (ensemble.RandomForestClassifier,
#          {
#              'model__n_estimators': [100, 50],
#              'model__criterion': ['gini', 'entropy']
#          }
#          )]

#     scalers = [
#         preprocessing.MinMaxScaler,
#         preprocessing.Normalizer,
#         preprocessing.StandardScaler,
#         preprocessing.RobustScaler
#     ]

#     scoringConfig = {"precision": metrics.make_scorer(metrics.precision_score, average="binary", zero_division=0),
#                      "recall": metrics.make_scorer(metrics.recall_score, average="binary", zero_division=0),
#                      "f1": metrics.make_scorer(metrics.f1_score, average="binary", zero_division=0),
#                      "accuracy": metrics.make_scorer(metrics.accuracy_score)
#                      }

#     pipes = generatePipelines(models, scalers)
#     fold = model_selection.StratifiedKFold(n_splits=10)
#     trainTestProcess(pipes, scoringConfig, "f1",
#                      0.2, fold, X, Y, "../results")
