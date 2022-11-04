from sklearn import ensemble, feature_selection, impute, linear_model, model_selection, neural_network, preprocessing, svm, metrics, neighbors, cluster
from imblearn import FunctionSampler, pipeline, under_sampling, over_sampling, combine
from nhanes_cvr import transformers as tf
from sklearn_extra.cluster import KMedoids


# CONFIGURATION VARIABLES
scoringConfig = {
    "precision": metrics.make_scorer(metrics.precision_score, average="binary", zero_division=0),
    "recall": metrics.make_scorer(metrics.recall_score, average="binary", zero_division=0),
    "f1": metrics.make_scorer(metrics.f1_score, average="binary", zero_division=0),
    "accuracy": metrics.make_scorer(metrics.accuracy_score),
    "auc_roc": metrics.make_scorer(metrics.roc_auc_score)
}

randomState = 999
splits = 10
fold = model_selection.StratifiedKFold(
    n_splits=splits, shuffle=True, random_state=randomState)
target = 'f1'
testSize = .2
maxIter = 200
folds = 5
foldRepeats = 10
testSize = .20
correlationThreshold = 0.05
zScoreThreshold = 2.9
nullThreshold = 3

kValues = [2, 3, 4]

clusterMethods = [cluster.KMeans, KMedoids]

bestScoresFunctions = [tf.highestScoreIndex,
                       tf.lowestScoreIndex,
                       tf.bestScoreByClosestToMean,
                       tf.bestScoreByClosestToMedian]

quickAllConfs = [{'k': k, 'findBest': s, 'clusterMethod': m}
                 for k in kValues for m in clusterMethods for s in bestScoresFunctions]

replacements = [
    impute.SimpleImputer(strategy='mean')
]

drops = [
    # preprocessing.FunctionTransformer(),
    tf.DropTransformer(threshold=0.5)
]

selections = [
    # feature_selection.SelectPercentile(),
    preprocessing.FunctionTransformer()
    # tf.CorrelationSelection(threshold=0.01)
]

scalers = [
    # preprocessing.FunctionTransformer(),
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
    (noSamplingPipeline, noSamplingConf),
    (samplerPipeline, samplerConf),
    (kusPipeline, kusPipelineConf),
    (kusWithSamplerPipeline, kusWithSamplerPipelineConf),
]
