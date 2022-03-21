from typing import List, Tuple
import pandas as pd
from sklearn import model_selection, decomposition, preprocessing
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

from utils import ensure_directory_exists, labelCauseOfDeathAsCVR, map_dataframe, process_dataset, scalableFeatures


def get_class_name(x: any) -> str:
    return x.__class__.__name__


def grid_search_train_model(model: any, params: any, X: pd.DataFrame, y: pd.Series,
                            cv: any, scoring: List[str], fit_score: str, save_dir: str):
    print(f"      on: {model}")
    modelName = get_class_name(model)

    gs = model_selection.GridSearchCV(estimator=model, param_grid=params,
                                      cv=cv, scoring=scoring, refit=fit_score,
                                      return_train_score=True, n_jobs=-1, pre_dispatch=10)

    res = gs.fit(X, y)

    pd.DataFrame(res.cv_results_).to_csv(
        f"{save_dir}/{modelName}_grid_search_res.csv")

    return model, res


def transform_to_csv_data(modelRes: Tuple[any, dict], scoring_res: List[str]) -> List[str]:
    m, res = modelRes
    best = res.best_index_
    scores = [res.cv_results_[x][best] for x in scoring_res]
    return [get_class_name(m)] + scores


def save_stats_for_cv_results(csvDataFrame, resDir, name, fit_score):
    x = csvDataFrame.model
    y = csvDataFrame[f"mean_test_{fit_score}"]

    plt.bar(x, y)
    plt.xticks(x, rotation=-15, fontsize="x-small")
    plt.xlabel("Model")
    plt.ylabel(f"Test {fit_score} Score")
    plt.title(name)
    plt.savefig(
        f"{resDir}/{name}_model_plot_results.png")
    plt.close()

    csvDataFrame.to_csv(
        f"{resDir}/{name}_model_results.csv")


def run_normalizer_cv_on_models(normalizer: any, cv: any, X: pd.DataFrame, Y: pd.Series, scalableList: List[str],
                                scoring: List[str], models: List[Tuple[any, any]], csv_columns: List[str], scoring_res: List[str], save_dir, fit_score) -> pd.DataFrame:
    normName = get_class_name(normalizer) if normalizer else "Normal"
    stratName = get_class_name(cv)
    k = cv.get_n_splits()
    fullName = f"{normName}_{k}_{stratName}"
    resDir = f"{save_dir}/{fullName}"

    ensure_directory_exists(resDir)

    print(f"Running {fullName}")

    normX = map_dataframe(lambda x: normalizer.fit_transform(
        x, Y), X, scalableList) if normalizer else X  # Scale X
    plot_pca(normX, Y, f"{save_dir}/{normName}_pca.png")

    modelCount = len(models)
    modelRes = [grid_search_train_model(
        m, p, normX, Y, cv, scoring, fit_score, resDir) for m, p in models]

    csv_data = [transform_to_csv_data(x, scoring_res) for x in modelRes]

    csv_dataframe = pd.DataFrame(
        csv_data, columns=csv_columns).assign(normalizer=pd.Series(normName, index=range(modelCount)),
                                              foldingStrat=pd.Series(stratName, index=range(modelCount)))

    save_stats_for_cv_results(csv_dataframe, resDir, fullName, fit_score)

    return csv_dataframe

# TODO Would like this to also turn back best models


def run_ml_pipeline(folding_strats, dataset, combine_directions, scoring, models, normalizers, csv_columns,         scoring_res, save_dir, run_name, fit_score) -> List[pd.DataFrame]:
    # Could do threading here but need to extract maplotlib functions out
    X, Y = process_dataset(dataset, combine_directions,
                           labelCauseOfDeathAsCVR)

    X.describe().to_csv(f"{save_dir}/{run_name}/feature_description.csv")
    X.hist(figsize=(10, 10))
    plt.savefig(f"{save_dir}/{run_name}/feature_hist.png")
    plt.close()

    X.corr().to_csv(f"{save_dir}/{run_name}/correlation_matrix.csv")
    X.corrwith(Y).to_csv(f"{save_dir}/{run_name}/correlation_to_y_matrix.csv")

    scalableList = scalableFeatures(combine_directions)

    save_dir = f"{save_dir}/{run_name}"
    ensure_directory_exists(save_dir)

    res = list()
    for normalizer in normalizers:
        for cv in folding_strats:
            x = run_normalizer_cv_on_models(
                normalizer, cv, X, Y, scalableList, scoring, models, csv_columns, scoring_res, save_dir, fit_score)
            res.append(x)

    res = pd.concat(res)

    plot_fold_results(res, save_dir, fit_score)
    plot3d_all_fold_results(res, save_dir, fit_score)
    res.to_csv(f'{save_dir}/all_results.csv')

    return res


def plot_pca(X: pd.DataFrame, Y: pd.Series, location: str):
    X_pca = decomposition.PCA().fit_transform(X)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(location)
    plt.close()


def plot_fold_results(res: pd.DataFrame, save_dir: str, fit_score: str):
    # Group Fold Results into Plot
    for foldName, data in res.groupby(['foldingStrat']):
        plt.title(foldName)
        for normName, data2 in data.groupby(['normalizer']):
            score = data2[f"mean_test_{fit_score}"]
            models = data2.model

            plt.scatter(models, score, label=normName)
        plt.xticks(models, rotation=-15, fontsize="x-small")
        plt.xlabel("Models")
        plt.ylabel(fit_score)
        plt.legend(loc="best")
        plt.savefig(f"{save_dir}/{foldName}_plot.png")
        plt.close()


def plot3d_all_fold_results(res: pd.DataFrame, save_dir: str, fit_score: str):
    # Display All Results in 3d plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    normEncoder = preprocessing.LabelEncoder()
    modelEncoder = preprocessing.LabelEncoder()
    res = res.assign(normalizer_enc=normEncoder.fit_transform(res.normalizer),
                     model_enc=modelEncoder.fit_transform(res.model))

    for foldName, data in res.groupby(['foldingStrat']):
        score = data[f"mean_test_{fit_score}"]
        models = data.model_enc
        norms = data.normalizer_enc

        ax.scatter(models, norms, score, label=foldName)

    plt.xticks(res.model_enc, modelEncoder.inverse_transform(res.model_enc))
    plt.yticks(res.normalizer_enc,
               normEncoder.inverse_transform(res.normalizer_enc))
    ax.set_xlabel("Models")
    ax.set_ylabel("Normalizations")
    ax.set_zlabel(fit_score)
    plt.legend(loc="best")
    plt.savefig(f"{save_dir}/all_results_3d_plot.png")
    plt.close()
