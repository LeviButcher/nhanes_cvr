from concurrent.futures import thread
import os
import threading
from typing import List, Tuple
import pandas as pd
from sklearn import model_selection, decomposition
import matplotlib.pyplot as plt
import numpy as np

SAVE_DIR = '../results'


def get_class_name(x: any) -> str:
    return x.__class__.__name__


def train_model(model: any, X: pd.DataFrame, y: pd.Series, cv: any, scoring: List[str]) -> Tuple[any, dict]:
    res = model_selection.cross_validate(
        model, X, y, cv=cv, scoring=scoring, return_train_score=True)

    return model, res


def transform_to_csv_data(modelRes: Tuple[any, dict], scoring_res: List[str]) -> List[str]:
    m, res = modelRes
    scores = [np.average(res[x]) for x in scoring_res]
    return [get_class_name(m)] + scores


def plot_pca_for_normalizer(X, Y, normalizer):
    normName = get_class_name(normalizer) if normalizer else "Normal"
    X[:] = normalizer.fit_transform(
        X, Y) if normalizer else X  # Scale X
    plot_pca(X, Y, f"../results/{normName}_pca.png")


def save_stats_for_cv_results(csvDataFrame, resDir, name):
    x = csvDataFrame.model
    y = csvDataFrame.test_f1

    plt.bar(x, y)
    plt.xticks(x, rotation=-15, fontsize="x-small")
    plt.xlabel("Model")
    plt.ylabel("Test F1 Score")
    plt.title(name)
    plt.savefig(
        f"{resDir}/{name}_model_plot_results.png")
    plt.close()

    csvDataFrame.to_csv(
        f"{resDir}/{name}_model_results.csv")


def run_normalizer_cv_on_models(normalizer: any, cv: any, X: pd.DataFrame, Y: pd.Series,
                                scoring: List[str], models: List[any], csv_columns: List[str], scoring_res: List[str]) -> pd.DataFrame:
    normName = get_class_name(normalizer) if normalizer else "Normal"
    stratName = get_class_name(cv)
    k = cv.get_n_splits()
    fullName = f"{normName}_{k}_{stratName}"
    resDir = f"{SAVE_DIR}/{fullName}"

    try:
        os.mkdir(resDir)
    except:
        pass

    print(f"Running {fullName}")

    X[:] = normalizer.fit_transform(
        X, Y) if normalizer else X  # Scale X
    plot_pca(X, Y, f"{SAVE_DIR}/{normName}_pca.png")

    modelCount = len(models)
    modelRes = [train_model(m, X, Y, cv, scoring) for m in models]
    csv_data = [transform_to_csv_data(
        x, scoring_res) for x in modelRes]

    csv_dataframe = pd.DataFrame(
        csv_data, columns=csv_columns).assign(normalizer=pd.Series(normName, index=range(modelCount)),
                                              foldingStrat=pd.Series(stratName, index=range(modelCount)))

    save_stats_for_cv_results(csv_dataframe, resDir, fullName)

    return csv_dataframe


def run_ml_pipeline(folding_strats, X, Y, scoring, models, normalizers, csv_columns, scoring_res) -> List[pd.DataFrame]:
    # Could do threading here but need to extract maplotlib functions out
    res = list()
    for normalizer in normalizers:
        for cv in folding_strats:
            x = run_normalizer_cv_on_models(
                normalizer, cv, X, Y, scoring, models, csv_columns, scoring_res)
            res.append(x)

    return pd.concat(res)


def plot_pca(X: pd.DataFrame, Y: pd.Series, location: str):
    X_pca = decomposition.PCA().fit_transform(X)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(location)
    plt.close()
