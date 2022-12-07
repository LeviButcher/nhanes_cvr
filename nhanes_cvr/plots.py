import pandas as pd
from sklearn import metrics
from typing import Hashable, Iterable, List
import matplotlib.pyplot as plt
import seaborn as sns
from nhanes_cvr import transformers as tf
from nhanes_cvr.types import *


def plotTestResults(results: ScoreDF, scoringDict: Scoring, title: str, savePath: str):
    scoring = list(scoringDict.keys())
    results = ScoreDF(results.astype(
        {'model': 'string', 'samplers': 'string'}))
    # g = sns.PairGrid(results, x_vars="samplers", hue="scaling")
    g = sns.scatterplot(data=results, x="model", y=scoring)

    # g.map(sns.scatterplot, "model", "f1")
    # g.tight_layout()
    plt.savefig(savePath)
    plt.close()


def plotConfusionMatrix(results: ScoreDF, title: str, savePath: str):
    g = sns.FacetGrid(results, row="model", col="scaling")

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


def plotPrecisionRecallForModels(models: Iterable[Tuple[Hashable, PipeLine]], X: pd.DataFrame, Y: pd.Series, savePath: str):
    ax = plt.gca()
    for i, m in models:
        metrics.PrecisionRecallDisplay.from_estimator(
            m, X, Y, ax=ax, name=f"best trained #{i}")

    ax.set_title("precision recall curve")
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center',
                    bbox_to_anchor=(1.3, 1))
    plt.savefig(savePath, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def plotROCCurve(models: Iterable[Tuple[Hashable, PipeLine]], X: pd.DataFrame, Y: pd.Series, savePath: str):
    _, ax = plt.subplots(1, sharex=True, sharey=True)
    for i, m in models:
        metrics.RocCurveDisplay.from_estimator(
            m, X, Y, ax=ax, name=f"best trained #{i}")

    plt.title("roc curve")
    handles, labels = ax.get_legend_handles_labels()  # type: ignore
    lgd = ax.legend(handles, labels, loc='upper center',  # type: ignore
                    bbox_to_anchor=(1.4, 1))
    plt.savefig(savePath, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def getBestF1ByModels(res: Union[ScoreDF, ScoreDF]):
    best = res.groupby("model")['f1'].idxmax()  # type: ignore
    return res.loc[best, :]


def runAllPlotting(gsResults: GridSearchDF, trainResults: ScoreDF, testResults: ScoreDF, bestModels: List[Tuple[str, PipeLine]],
                   trainX: pd.DataFrame, trainY: pd.Series, testX: pd.DataFrame, testY: pd.Series,
                   scoring: Scoring, saveDir: str):

    # Save CSVs
    trainResults.to_csv(f"{saveDir}train_results.csv")
    gsResults.to_csv(f"{saveDir}gs_results.csv")
    testResults.to_csv(f"{saveDir}test_results.csv")

    # Training Plots
    bestModelsDF = pd.DataFrame(bestModels, columns=["name", "pipeline"])
    for n, df in bestModelsDF.groupby(by="name"):
        plotPrecisionRecallForModels(
            df.pipeline.items(), trainX, trainY, f"{saveDir}{n}_train_precision_recall")
        plotROCCurve(df.pipeline.items(), trainX, trainY,
                     f"{saveDir}{n}_train_roc_curve")

    # Test Plots
    for n, df in bestModelsDF.groupby(by="name"):
        plotPrecisionRecallForModels(
            df.pipeline.items(), testX, testY, f"{saveDir}{n}_test_precision_recall")
        plotROCCurve(df.pipeline.items(), testX, testY,
                     f"{saveDir}{n}_test_roc_curve")


def correlationAnalysis(X: pd.DataFrame, y: pd.Series, saveDir: str):
    corr = X.corrwith(y).abs().sort_values(ascending=False)
    marker = corr.quantile(q=.9)
    res = corr[corr >= marker]

    corr.to_csv(f"{saveDir}correlation_analysis.csv")
    res.to_csv(f"{saveDir}correlation_10th_percentile.csv")


def saveChosenFeatures(pipeline: PipeLine, trainX: pd.DataFrame, savePath: str):
    # cols = trainX.columns
    # scores = pipeline.named_steps['selection'].scores_
    # print(pipeline.named_steps['selection'].feature_names_in_)
    # res = pipeline.transform(trainX)
    # res = pd.Series(scores, index=cols)
    # print(res)

    # bestFeatures.to_csv(f"{saveDir}chosen_features.csv")
    return
