import imblearn
from sklearn import ensemble, linear_model, neural_network, svm, metrics, neighbors, cluster
import numpy as np
from imblearn import FunctionSampler
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
maxIter = 200
randomState = np.random.RandomState(0)
folds = 10
foldRepeats = 10
testSize = .20
correlationThreshold = 0.05
zScoreThreshold = 2.9
nullThreshold = 3

# These our hyperparamters
kValues = [2, 3, 4]

clusterMethods = [cluster.KMeans, KMedoids]

bestScoresFunctions = [tf.highestScoreIndex,
                       tf.lowestScoreIndex,
                       tf.bestScoreByClosestToMean,
                       tf.bestScoreByClosestToMedian]

quickAllConfs = [{'k': k, 'findBest': s, 'clusterMethod': m}
                 for k in kValues for m in clusterMethods for s in bestScoresFunctions]

# Going to want to seperate each sampler by type to easily combine later
samplers = [
    # (imblearn.under_sampling.RandomUnderSampler, {}),
    # (imblearn.under_sampling.ClusterCentroids, {}),
    # (imblearn.over_sampling.RandomOverSampler, {}),
    # (imblearn.over_sampling.SMOTE, {}),
    # (imblearn.combine.SMOTEENN, {}),
    # (imblearn.combine.SMOTETomek, {}),
    # TODO: Make it to where this can keep the KUS name for the sampler
    (lambda: FunctionSampler(func=tf.kMeansUnderSampling), {
        'sampler__kw_args': quickAllConfs
    })
]


models = [
    (linear_model.LogisticRegression, {}),
    # ((ensemble.RandomForestClassifier), {
    #  'model__random_state': [randomState]}),
    # (neural_network.MLPClassifier, {}),
    (svm.LinearSVC, {}),
    # (neighbors.KNeighborsClassifier, {}),
    # (neural_network.MLPClassifier, {}),
]


def combineModelAndSamplers(modelConf, samplerConf):
    model, mConf = modelConf
    sampler, sConf = samplerConf

    conf = {}
    for k, v in mConf.items():
        conf[f"model__{k}"] = v

    for k, v in sConf.items():
        conf[f"model__{k}"] = v

    # Hard part will be how to combine dictionary...

    pipeline = imblearn.pipeline.Pipeline(
        [('sampler', sampler()), ('model', model())])

    return (lambda: pipeline, conf)


pipelineModels = [combineModelAndSamplers(mc, sc)
                  for mc in models for sc in samplers]

# allModels = models + pipelineModels
allModels = pipelineModels
