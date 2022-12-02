import pandas as pd
from sklearn import ensemble, impute, linear_model, model_selection, neural_network, preprocessing, svm, metrics, neighbors, cluster
from imblearn import FunctionSampler, pipeline, under_sampling, over_sampling, combine
from sklearn_extra.cluster import KMedoids
from nhanes_cvr import transformers as tf
import nhanes_cvr.utils as utils
import nhanes_cvr.mlProcess as ml


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
testSize = .20
kValues = [2, 3, 4]
clusterMethods = [cluster.KMeans, KMedoids]
bestScoresFunctions = [tf.highestScoreIndex,
                       tf.lowestScoreIndex,
                       tf.bestScoreByClosestToMean,
                       tf.bestScoreByClosestToMedian]

# All configs for kMeansUnderSampling
quickAllConfs = [{'k': k, 'findBest': s, 'clusterMethod': m}
                 for k in kValues for m in clusterMethods for s in bestScoresFunctions]

replacements = [
    impute.SimpleImputer(strategy='mean')
]

drops = [
    preprocessing.FunctionTransformer(),
]

selections = [
    preprocessing.FunctionTransformer()
]

scalers = [
    preprocessing.MinMaxScaler(),
    preprocessing.StandardScaler(),
]

underSamples = [
    under_sampling.RandomUnderSampler(),
    under_sampling.ClusterCentroids(),
]


overSamplers = [
    over_sampling.RandomOverSampler(),
    over_sampling.SMOTE(),
]

combineSamplers = [
    combine.SMOTEENN(),
    combine.SMOTETomek(),
]

samplers = underSamples + overSamplers + combineSamplers


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

noSamplingPipeline = pipeline.Pipeline([
    ('drop', drops[0]),
    ('replacement', replacements[0]),
    ('scaling', scalers[0]),
    ('selection', selections[0]),
    ("model", models[0])
])

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

kusPipelineConf = noSamplingConf | {
    'kus__kw_args': quickAllConfs,
}

kusPipeline = pipeline.Pipeline([
    ('drop', drops[0]),
    ('replacement', replacements[0]),
    ('scaling', scalers[0]),
    ('selection', selections[0]),
    ('kus', FunctionSampler(func=tf.kMeansUnderSampling)),
    ("model", models[0])
])

kusWithSamplerPipelineConf = samplerConf | kusPipelineConf

kusWithSamplerPipeline = pipeline.Pipeline([
    ('drop', drops[0]),
    ('replacement', replacements[0]),
    ('scaling', scalers[0]),
    ('selection', selections[0]),
    ('kus', FunctionSampler(func=tf.kMeansUnderSampling)),
    ('samplers', samplers[0]),
    ("model", models[0])
])


allPipelines = [
    ("noSampling", noSamplingPipeline, noSamplingConf),
    ("withSampling", samplerPipeline, samplerConf),
    ("kus", kusPipeline, kusPipelineConf),
    ("kusWithSampling", kusWithSamplerPipeline, kusWithSamplerPipelineConf),
]

labelMethods = [
    ("hypertension_paper", utils.nhanesToHypertensionPaperSet),
]

getRiskFunctions = [
    ("cvrDeath", utils.nhanesCVRDeath),
    ("heartFailure", utils.nhanesHeartFailure),
]


def runHypertensionRiskAnalyses(dataset: pd.DataFrame, saveDir: str):
    utils.makeDirectoryIfNotExists(f"{saveDir}")

    for n, f in labelMethods:
        X, Y = f(dataset)
        utils.makeDirectoryIfNotExists(f"{saveDir}/{n}")
        X.describe().to_csv(f"{saveDir}/{n}/dataset_info.csv")
        Y.value_counts().to_csv(f"{saveDir}/{n}/label_info.csv")
        X.dtypes.to_csv(f"{saveDir}/{n}/dataset_types.csv")

    # Turn into run risk analyses
    ml.runRiskAnalyses("hypertensionAllRisk", labelMethods, allPipelines, scoringConfig,
                       target, testSize, fold, dataset, getRiskFunctions, saveDir)
