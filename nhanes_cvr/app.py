from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import feature_selection, linear_model, metrics, model_selection, preprocessing, impute
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import nhanes_cvr.utils as utils
import seaborn as sns
from nhanes_cvr.config import dataset, testSize, scoringConfig, models, scalers, randomState
import nhanes_cvr.mlProcess as ml
from imblearn import under_sampling, combine, pipeline
from nhanes_cvr.transformers import CorrelationSelection, DropTransformer

# Matplotlib/Seaborn Theming
sns.set_theme(style='darkgrid', palette='pastel')

dataset.dtypes.to_csv("results/feature_types.csv")

labelMethods = [
    ("questionnaire", utils.nhanesToQuestionnaireSet),
    ("lab_thresh", utils.labelCVRBasedOnLabMetrics(2)),
    ("cardiovascular_codebook", utils.labelCVRBasedOnCardiovascularCodebook),
    # ("cvr_death", utils.nhanesToMortalitySet),
    # ("cvr_death_extra", utils.labelCVrBasedOnNHANESMortalityAndExtraFactors)
]


splits = 10
fold = model_selection.StratifiedKFold(
    n_splits=splits, shuffle=True, random_state=randomState)
target = 'f1'
testSize = .2

replacements = [
    lambda: impute.SimpleImputer(strategy='most_frequent')
]

selections = [
    # lambda: feature_selection.SelectPercentile()
    lambda: CorrelationSelection(threshold=0.01)
]

keep = (dataset.dtypes == 'float64')
dataset = dataset.loc[:, keep]

pcaPipe = pipeline.make_pipeline(DropTransformer(threshold=0.5),
                                 impute.SimpleImputer(),
                                 preprocessing.StandardScaler(),
                                 PCA(n_components=2))

for n, f in labelMethods:
    X, Y = f(dataset)
    X = pcaPipe.fit_transform(X, Y)
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.savefig(f"results/{n}_pca.png")


# No Sampling

# cvModels = ml.generatePipelines(
#     models, scalers, replacements, selections)  # type: ignore

# for nl in labelMethods:
#     ml.labelThenTrainTest(nl, cvModels, scoringConfig, target,  # type: ignore
#                           testSize, fold, dataset, "results/no_sampling")

# SMOTETOMEK

# cvModels = ml.generatePipelinesWithSampling(
#     models, scalers, replacements, [lambda: combine.SMOTETomek()], selections)  # type: ignore

# for nl in labelMethods:
#     ml.labelThenTrainTest(nl, cvModels, scoringConfig, target,  # type: ignore
#                           testSize, fold, dataset, "results/smotetomek")

# SMOTEENN

# cvModels = ml.generatePipelinesWithSampling(
#     models, scalers, replacements, [lambda: combine.SMOTEENN()], selections)  # type: ignore

# for nl in labelMethods:
#     ml.labelThenTrainTest(nl, cvModels, scoringConfig, target,  # type: ignore
#                           testSize, fold, dataset, "results/smoteenn")

cvModels = ml.generatePipelinesWithSampling(
    models, scalers, replacements, [lambda: combine.SMOTEENN()], [lambda: feature_selection.RFE(  # type: ignore
        estimator=RandomForestClassifier(), step=.3)])

for nl in labelMethods:
    ml.labelThenTrainTest(nl, cvModels, scoringConfig, target,  # type: ignore
                          testSize, fold, dataset, "results/best")
