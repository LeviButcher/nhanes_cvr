import pandas as pd
from sklearn import metrics
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
from nhanes_cvr.types import *
from nhanes_cvr import utils


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
    g = sns.FacetGrid(results, row="modelAppr", col="scalingAppr")

    # hackish way of getting heatmap drawn
    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop('data')
        sns.heatmap(data.cm[0], **kwargs)

    g.map_dataframe(draw_heatmap, data="cm",
                    annot=True, fmt="d", square=True, cbar=False, cmap="Blues")

    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(title)
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.fig.tight_layout()

    plt.savefig(savePath)
    plt.close()


def plotPrecisionRecallForModels(models: List[CVSearch], X: pd.DataFrame, Y: pd.Series, savePath: str):
    ax = plt.gca()
    for m in models:
        modelName, scalerName, *_ = utils.getPipelineClassesAppr(m)
        metrics.PrecisionRecallDisplay.from_estimator(
            m, X, Y, ax=ax, name=f"{modelName} - {scalerName}")

    ax.set_title("precision recall curve")
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center',
                    bbox_to_anchor=(1.3, 1))
    plt.savefig(savePath, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def plotROCCurve(models: List[CVSearch], X: pd.DataFrame, Y: pd.Series, savePath: str):
    _, ax = plt.subplots(1, sharex=True, sharey=True)
    for m in models:
        modelName, scalerName, *_ = utils.getPipelineClassesAppr(m)
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


def randomForestFeatureImportance(models: List[CVSearch], features: List[str], savePath: str):
    rfModels = [m for m in models
                if utils.getPipelineClasses(m)[0] == "RandomForestClassifier"]

    results = []
    for rf in rfModels:
        imp = rf.best_estimator_['model'].feature_importances_  # type: ignore
        data = pd.DataFrame([imp], columns=features) \
            .assign(model='RandomForestClassifier') \
            .assign(scaling=rf.best_estimator_['scaling'])  # type: ignore
        results.append(data)
    results = pd.concat(results, ignore_index=True)
    results.to_csv(f"{savePath}")


def getBestF1ByModels(res: Union[CVTestDF, CVTestDF]):
    best = res.groupby("model")['f1'].idxmax()
    return res.loc[best, :]


def runAllPlotting(trainResults: CVTrainDF, testResults: CVTestDF, trainedModels: List[CVSearch],
                   trainX: pd.DataFrame, trainY: pd.Series, testX: pd.DataFrame, testY: pd.Series,
                   scoring: Scoring, saveDir: str):
    scoringNames = list(scoring.keys())

    trainResults.to_csv(f"{saveDir}train_results.csv")
    plotValCurveForModels(trainResults, 10, f"{saveDir}train_val_curves")

    # features = trainX.columns.to_list()
    # randomForestFeatureImportance(
    #     trainedModels, features, f"{saveDir}random_forest_importance.csv")

    # Training Plots
    plotPrecisionRecallForModels(
        trainedModels, trainX, trainY, f"{saveDir}train_precision_recall")
    plotROCCurve(trainedModels, trainX, trainY, f"{saveDir}train_roc_curve")

    # Test Plots
    plotTestResults(testResults, scoringNames,
                    f"{saveDir} - test scores", f"{saveDir}test_scores")
    plotPrecisionRecallForModels(
        trainedModels, testX, testY, f"{saveDir}test_precision_recall")
    plotROCCurve(trainedModels, testX, testY, f"{saveDir}test_roc_curve")
    # plotConfusionMatrix(testResults, "test - CM",
    #                     f"{saveDir}test_confusion_matrix")

    testResults.to_csv(f"{saveDir}test_results.csv")
    res = getBestF1ByModels(testResults)
    res.to_csv(f"{saveDir}best_test_results.csv")
