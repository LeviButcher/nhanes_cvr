import pandas as pd
from sklearn import preprocessing, model_selection, linear_model
from sklearn.model_selection import GridSearchCV
from typing import Any, Dict, List, NamedTuple, NewType, Tuple, Union
from functools import reduce
from nhanes_cvr.utils import getClassName

# Sucks to have to type each one out but it makes this type safe
Model = linear_model.LinearRegression
Scaler = Union[preprocessing.MaxAbsScaler, preprocessing.MinMaxScaler]
Folding = Union[model_selection.KFold,  model_selection.StratifiedKFold]
ModelParams = Dict[str, List[Any]]
Scoring = Dict[str, Any]
Target = str
FittedGridSearch = NewType('FittedGridSearch', GridSearchCV)
Results = NewType('Results', pd.DataFrame)


class ScalerConfig(NamedTuple):
    scaler: Union[Scaler, None]
    features: List[str]


def emptyConfig(scaler: Scaler) -> ScalerConfig:
    return ScalerConfig(scaler, [])


def createScalerConfigs(scalers: List[Scaler], features: List[str]) -> List[ScalerConfig]:
    return [ScalerConfig(s, features) for s in scalers]


def createScalerConfigsIgnoreFeatures(scalers: List[Scaler], X: pd.DataFrame, features: List[str]) -> List[ScalerConfig]:
    return [withoutScaling(emptyConfig(s), X, features) for s in scalers]


def addFeature(config: ScalerConfig, feature: str) -> ScalerConfig:
    return ScalerConfig(config.scaler, config.features + [feature])


def addFeatures(config: ScalerConfig, features: List[str]) -> ScalerConfig:
    return reduce(addFeature, features, config)


def withoutScaling(config: ScalerConfig, X: pd.DataFrame, features: List[str]):
    toScale = [x for x in X.columns if x not in features]
    return addFeatures(config, toScale)


def runScaling(config: ScalerConfig, X: pd.DataFrame, Y: pd.Series) -> pd.DataFrame:
    # Run the scaler on the feature, making a new dataframe of scaled features
    res = X.copy()
    if config.scaler is None:
        return res

    scaled = config.scaler.fit_transform(X.loc[:, config.features])
    res.loc[:, config.features] = scaled

    return res


class GridSearchConfig(NamedTuple):
    model: Model
    modelParams: ModelParams
    scalerConfig: ScalerConfig
    folding: Folding
    scoring: Scoring

    def __str__(self):
        return f"{getClassName(self.model)} - {getClassName(self.scalerConfig.scaler)} - {getClassName(self.folding)}"


def runGridSearch(config: GridSearchConfig, target: str, X: pd.DataFrame, Y: pd.Series) -> FittedGridSearch:
    print(f"Run Grid Search -> {config}")

    # Don't really care about scaling multiple times... for now
    scaled = runScaling(config.scalerConfig, X, Y)

    res = GridSearchCV(
        estimator=config.model, param_grid=config.modelParams, cv=config.folding,
        refit=target, scoring=config.scoring, return_train_score=True,
        n_jobs=10)

    return res.fit(scaled, Y)


def runMultipleGridSearchs(configs: List[GridSearchConfig], target: str, X: pd.DataFrame, Y: pd.Series) -> List[Tuple[GridSearchConfig, FittedGridSearch]]:
    return [(c, runGridSearch(c, target, X, Y)) for c in configs]


def runMultipleGridSearchAsync(configs: List[GridSearchConfig], target: str, X: pd.DataFrame, Y: pd.Series, ) -> List[Tuple[GridSearchConfig, FittedGridSearch]]:
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        return list(zip(configs, executor.map(
            lambda c: runGridSearch(c, target, X, Y), configs)))


def createGridSearchConfigs(modelConfigs: List[Tuple[Model, ModelParams]],
                            scalers: List[ScalerConfig], foldings: List[Folding],
                            scoringList: List[Scoring]) -> List[GridSearchConfig]:

    return [GridSearchConfig(m, mp, s, f, scoring)
            for (m, mp) in modelConfigs
            for s in scalers
            for f in foldings
            for scoring in scoringList
            ]


def resultsStoredForConfig(config: GridSearchConfig) -> List[str]:
    """
    Returns List of all names of results stored in FittedGridSearch
    """
    types = ["test", "train"]

    return [f"mean_{x}_{y}" for x in types for y in config.scoring]


def resultToDataFrame(config: GridSearchConfig, res: FittedGridSearch) -> Results:
    best = res.best_index_

    names = resultsStoredForConfig(config)
    runInfo = [getClassName(x)
               for x in [config.model, config.scalerConfig.scaler, config.folding]]
    row = runInfo + [res.cv_results_[x][best] for x in names]
    columns = ["model", "scaler", "folding"] + names

    return Results(pd.DataFrame([row], columns=columns))


def resultsToDataFrame(results: List[Tuple[GridSearchConfig, FittedGridSearch]]) -> Results:
    res = [resultToDataFrame(config, x) for config, x in results]
    return Results(pd.concat(res, ignore_index=True))


def plotResults3d(res: Results, score: str):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    normEncoder = preprocessing.LabelEncoder()
    modelEncoder = preprocessing.LabelEncoder()
    res = Results(res.assign(scaler_enc=normEncoder.fit_transform(res.scaler),
                             model_enc=modelEncoder.fit_transform(res.model)))

    for foldName, data in res.groupby(['folding']):
        scores = data[f"mean_test_{score}"]
        ax.scatter(data.model_enc, data.scaler_enc, scores, label=foldName)

    plt.xticks(res.model_enc, modelEncoder.inverse_transform(res.model_enc))
    plt.yticks(res.scaler_enc, normEncoder.inverse_transform(res.scaler_enc))
    ax.set_xlabel("Models")
    ax.set_ylabel("Scaling")
    ax.set_zlabel(score)
    plt.legend(loc="best")
    plt.savefig(f"./results/all_results_3d_plot.png")
    plt.close()


def get_best(results: List[Tuple[GridSearchConfig, FittedGridSearch]]) -> Tuple[GridSearchConfig, FittedGridSearch]:

    return results[0]
