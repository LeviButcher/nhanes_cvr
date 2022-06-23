import pandas as pd
from sklearn import decomposition, metrics, preprocessing, model_selection, linear_model, cluster
from sklearn.model_selection import GridSearchCV
from typing import Any, Dict, List, NamedTuple, NewType, Tuple, Union, Callable
from functools import reduce
from nhanes_cvr.utils import getClassName
import seaborn as sns
import matplotlib.pyplot as plt
import nhanes_cvr.utils as utils
from toolz import curry

# Sucks to have to type each one out but it makes this type safe
Model = Callable[[], linear_model.LinearRegression]
Scaler = Union[preprocessing.MaxAbsScaler, preprocessing.MinMaxScaler]
Folding = Union[model_selection.KFold,  model_selection.StratifiedKFold]
ModelParams = Dict[str, List[Any]]
Scoring = Dict[str, Any]
Target = str
FittedGridSearch = NewType('FittedGridSearch', GridSearchCV)
Results = NewType('Results', pd.DataFrame)
ModelWithParams = Tuple[Model, ModelParams]


class ScalerConfig(NamedTuple):
    scaler: Union[Scaler, None]
    features: List[str]


def emptyConfig(scaler: Scaler) -> ScalerConfig:
    return ScalerConfig(scaler, [])


def createScalerConfigs(scalers: List[Scaler], features: List[str]) -> List[ScalerConfig]:
    return [ScalerConfig(s, features) for s in scalers]


def getScalerName(sc: ScalerConfig) -> str:
    return utils.getClassName(sc.scaler)


@curry
def createScalerConfigsIgnoreFeatures(scalers: List[Scaler], features: List[str],  X: pd.DataFrame) -> List[ScalerConfig]:
    return [withoutScaling(emptyConfig(s), X, features) for s in scalers]


@curry
def createScalerAllFeatures(scalers: List[Scaler], X: pd.DataFrame):
    return [addFeatures(emptyConfig(s), list(X.columns)) for s in scalers]


def addFeature(config: ScalerConfig, feature: str) -> ScalerConfig:
    return ScalerConfig(config.scaler, config.features + [feature])


def addFeatures(config: ScalerConfig, features: List[str]) -> ScalerConfig:
    return reduce(addFeature, features, config)


def withoutScaling(config: ScalerConfig, X: pd.DataFrame, features: List[str]):
    toScale = [x for x in X.columns if x not in features]
    return addFeatures(config, toScale)


def runScaling(config: ScalerConfig, X: pd.DataFrame, Y: pd.Series) -> pd.DataFrame:
    # Run the scaler on the feature, making a new dataframe of scaled features
    # NOTE: Make SURE not to modify original X or Y
    res = X.copy()
    if config.scaler is None:
        return res

    scaled = config.scaler.fit_transform(X.loc[:, config.features])
    res.loc[:, config.features] = scaled

    return res


class GridSearchConfig(NamedTuple):
    model: Model
    modelParams: ModelParams
    folding: Folding
    scoring: Scoring

    def __str__(self):
        return f"{getClassName(self.model())} - {getClassName(self.folding)}"


GridSearchRun = Tuple[GridSearchConfig, FittedGridSearch]


def runGridSearch(config: GridSearchConfig, target: str, X: pd.DataFrame, Y: pd.Series) -> FittedGridSearch:
    print(f"Run Grid Search -> {config}")

    res = GridSearchCV(
        estimator=config.model(), param_grid=config.modelParams, cv=config.folding,
        refit=target, scoring=config.scoring, return_train_score=True,
        n_jobs=10)

    return res.fit(X, Y)


def runMultipleGridSearches(configs: List[GridSearchConfig], target: str, X: pd.DataFrame, Y: pd.Series) -> List[GridSearchRun]:
    return [(c, runGridSearch(c, target, X, Y)) for c in configs]


def runMultipleGridSearchesAsync(configs: List[GridSearchConfig], target: str, X: pd.DataFrame, Y: pd.Series, ) -> List[GridSearchRun]:
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        return list(zip(configs, executor.map(
            lambda c: runGridSearch(c, target, X, Y), configs)))


def createGridSearchConfigs(modelConfigs: List[ModelWithParams], foldings: List[Folding],
                            scoringList: List[Scoring]) -> List[GridSearchConfig]:

    return [GridSearchConfig(m, mp, f, scoring)
            for (m, mp) in modelConfigs
            for f in foldings
            for scoring in scoringList
            ]


def resultsScores(result: FittedGridSearch) -> List[str]:
    """
    Returns List of all names of results stored in FittedGridSearch
    """
    types = ["test", "train"]

    return [f"mean_{t}_{score}" for t in types for score in list(result.scorer_)]


def getTestScoreNames(scoringConfig) -> List[str]:
    types = ["train", "test"]
    return [f"mean_{t}_{s}" for s in scoringConfig.keys() for t in types]


def resultToDataFrame(config: GridSearchConfig, res: FittedGridSearch) -> Results:
    best = res.best_index_
    # TODO add scoring method to res
    # TODO add best params string to res
    names = resultsScores(res)
    runInfo = [getClassName(x)
               for x in [config.model(), config.folding]]
    row = runInfo + [res.cv_results_[x][best] for x in names]
    columns = ["model", "folding"] + names

    return Results(pd.DataFrame([row], columns=columns))


def resultsToDataFrame(results: List[Tuple[GridSearchConfig, FittedGridSearch]]) -> Results:
    res = [resultToDataFrame(config, x) for config, x in results]
    return Results(pd.concat(res, ignore_index=True))


def plotResults3d(res: Results, score: str, savePath: str):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    modelEncoder = preprocessing.LabelEncoder().fit(res.model)
    scalerEncoder = preprocessing.LabelEncoder().fit(res.scaler)

    for foldName, data in res.groupby("folding"):
        x = modelEncoder.transform(data.model)
        y = scalerEncoder.transform(data.scaler)
        z = data[score]
        ax.scatter(x, y, z, label=foldName)

    plt.xticks(modelEncoder.transform(res.model),
               res.model, fontsize=6)
    plt.yticks(scalerEncoder.transform(res.scaler),
               res.scaler, fontsize=6)
    ax.set_xlabel("Models")
    ax.set_ylabel("Scaling")
    ax.set_zlabel(score)

    plt.legend(loc="best")
    plt.savefig(savePath)
    plt.close()


def getBestResult(results: List[GridSearchRun]) -> GridSearchRun:
    def compare(a: GridSearchRun, b: GridSearchRun):
        return a if a[1].best_score_ > b[1].best_score_ else b

    return reduce(compare, results)


def printResult(result: GridSearchRun) -> None:
    conf, res = result
    scores = resultsScores(res)
    scoreValues = [res.cv_results_[x][res.best_index_] for x in scores]

    print(f"Config = {conf}")
    print(f"Best Score: {res.best_score_} using {res.refit}")
    print(list(zip(scores, scoreValues)))


def printBestResult(results: List[GridSearchRun]):
    best = getBestResult(results)
    printResult(best)


def plotResultsGroupedByModel(res: Results, score: str, savePath: str, title=""):
    # NOTE: Bunches together folding method names
    # Not a big deal for now
    completePath = f"{savePath}_{score}"
    grid = sns.FacetGrid(res, col="model", hue="scaler",
                         col_wrap=3, legend_out=True)
    grid.map(sns.scatterplot, "folding", score)
    grid.add_legend()
    grid.fig.subplots_adjust(top=0.8)
    grid.fig.suptitle(title, fontsize=16)

    grid.savefig(completePath)
    plt.close()


def evaluateModel(testX: pd.DataFrame, testY: pd.Series, scoringConfig: Scoring,  gs: GridSearchRun) -> Results:
    config, model = gs
    scoreValues = [f(model, testX, testY) for (_, f) in scoringConfig.items()]
    scoreNames = [n for n in scoringConfig.keys()]
    runInfo = [getClassName(x)
               for x in [config.model(), config.folding]]
    row = runInfo + scoreValues
    columns = ["model", "folding"] + scoreNames

    return Results(pd.DataFrame([row], columns=columns))


def evaluateModels(testX: pd.DataFrame, testY: pd.Series, scoringConfig: Scoring, gridSearches: List[GridSearchRun]) -> Results:
    return Results(pd.concat([evaluateModel(testX, testY, scoringConfig, gs) for gs in gridSearches]))


# Could use better name
def runAndEvaluateGridSearch(X: pd.DataFrame, Y: pd.Series, testSize: float,
                             randomState: int, scoringConfig: Scoring,
                             models: List[ModelWithParams],
                             foldingStrategies: List[Folding], targetScore, savePath):
    print(f"X: {X.shape}")
    print(f"Y: {Y.shape}")

    trainX, testX, trainY, testY = model_selection.train_test_split(
        X, Y, test_size=testSize, random_state=randomState, stratify=Y)

    # Create Configs
    gridSearchConfigs = createGridSearchConfigs(
        models, foldingStrategies, [scoringConfig])

    # Run Training
    res = runMultipleGridSearches(gridSearchConfigs, targetScore,
                                  trainX, trainY)
    trainResultsDF = resultsToDataFrame(res)

    # Output Best Info
    print("\n--- FINISHED ---\n")
    print(f"Ran {len(res)} Configs")
    printBestResult(res)

    testResultsDF = evaluateModels(testX, testY, scoringConfig, res)

    results = trainResultsDF.set_index(["model", "folding"]).join(testResultsDF.set_index(
        ["model", "folding"]), how="inner", lsuffix="_train", rsuffix="_test")

    plotROCAUCForModels(res, testX, testY, f"{savePath}/rocauc")
    plotPrecisionRecallCurvesForModels(
        res, testX, testY, f"{savePath}/precision_recall_curve")

    return results.reset_index()


def plotROCAUCForModels(models: List[GridSearchRun], X, Y, savePath):
    for conf, m in models:
        modelName = utils.getClassName(conf.model())
        ax = plt.gca()
        metrics.RocCurveDisplay.from_estimator(
            estimator=m, X=X, y=Y, name=modelName, ax=ax)
    plt.savefig(f"{savePath}")
    plt.close()


def plotPrecisionRecallCurvesForModels(models: List[GridSearchRun], X, Y, savePath):
    for conf, m in models:
        modelName = utils.getClassName(conf.model())
        ax = plt.gca()
        metrics.PrecisionRecallDisplay.from_estimator(
            estimator=m, X=X, y=Y, name=modelName, ax=ax)
    plt.savefig(f"{savePath}")
    plt.close()
