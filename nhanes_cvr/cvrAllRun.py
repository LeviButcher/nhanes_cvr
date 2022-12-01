import pandas as pd
from sklearn import ensemble, impute, linear_model, model_selection, neural_network, preprocessing, svm, metrics, neighbors, preprocessing, feature_selection
from imblearn import FunctionSampler, pipeline, under_sampling, combine
import sklearn
from sklearn.compose import make_column_selector
from nhanes_cvr import transformers as tf
import nhanes_cvr.utils as utils
import nhanes_cvr.mlProcess as ml
from sklearn_pandas import DataFrameMapper


# CONFIGURATION VARIABLES
scoringConfig = {
    "precision": metrics.make_scorer(metrics.precision_score, average="binary", zero_division=0),
    "recall": metrics.make_scorer(metrics.recall_score, average="binary", zero_division=0),
    "f1": metrics.make_scorer(metrics.f1_score, average="binary", zero_division=0),
    "accuracy": metrics.make_scorer(metrics.accuracy_score),
    "auc_roc": metrics.make_scorer(metrics.roc_auc_score)
}

randomState = 42
splits = 10
fold = model_selection.StratifiedKFold(
    n_splits=splits, shuffle=True, random_state=randomState)
target = 'f1'
testSize = .2


replacements = [
    impute.SimpleImputer(strategy='mean')
]

drops = [
    tf.DropTransformer(threshold=0.5)
]

selections = [
    feature_selection.SelectPercentile(
        tf.correlationScore, percentile=10)  # type: ignore
]

scalers = [
    preprocessing.MinMaxScaler(),
    preprocessing.StandardScaler(),
]

samplers = [
    under_sampling.RandomUnderSampler(),
    # combine.SMOTEENN()
]


models = [
    linear_model.LogisticRegression(),
    ensemble.RandomForestClassifier(random_state=randomState),
    svm.LinearSVC(),
    neighbors.KNeighborsClassifier(),
    neural_network.MLPClassifier(max_iter=200),
]

noSamplingConf = {
    'drop': drops,
    'replacement': replacements,
    'scaling': scalers,
    'selection': selections,
    "model": models
}

samplerConf = noSamplingConf | {
    'samplers': samplers,
}

samplerPipeline = pipeline.Pipeline([
    ('drop', drops[0]),
    ('replacement', replacements[0]),
    ('scaling', scalers[0]),
    ('selection', selections[0]),
    ('samplers', samplers[0]),
    ("model", models[0])
])

allPipelines = [
    (samplerPipeline, samplerConf)
]

labelMethods = [
    ("lab_thresh", utils.nhanesToLabSet(2)),
    ("cardiovascular_codebook", utils.nhanesToCardiovascularCodeBookSet),
    ("questionnaire", utils.nhanesToQuestionnaireSet),
    ("cvr_death", utils.nhanesToMortalitySet),
]


def runCVRAllRiskAnalyses(dataset: pd.DataFrame, saveDir: str):
    utils.makeDirectoryIfNotExists(f"{saveDir}")

    for n, f in labelMethods:
        X, Y = f(dataset)
        print(X.shape)
        utils.makeDirectoryIfNotExists(f"{saveDir}/{n}")
        X.describe().to_csv(f"{saveDir}/{n}/dataset_info.csv")
        Y.value_counts(normalize=False).to_csv(f"{saveDir}/{n}/label_info.csv")
        X.dtypes.to_csv(f"{saveDir}/{n}/dataset_types.csv")

    # Turn into run risk analyses
    # ml.runRiskAnalyses("cvrAllRisk", labelMethods, allPipelines, scoringConfig, target,
    #                    testSize, fold, dataset, utils.nhanesCVRDeath, saveDir)
