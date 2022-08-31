from matplotlib import pyplot as plt
import pandas as pd
from sklearn import feature_selection, linear_model, metrics, model_selection, preprocessing, impute, datasets
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import nhanes_cvr.utils as utils
import seaborn as sns
import nhanes_cvr.mlProcess as ml
from imblearn import under_sampling, combine, pipeline, FunctionSampler
from nhanes_cvr.transformers import CorrelationSelection, DropTransformer, DropNullsTransformer, iqrBinaryClassesRemoval
import nhanes_cvr.transformers as trans
from nhanes_cvr.config import testSize, scoringConfig, models, scalers, randomState


dataset = utils.get_nhanes_dataset()

# Matplotlib/Seaborn Theming
sns.set_theme(style='darkgrid', palette='pastel')

dataset.dtypes.to_csv("results/feature_types.csv")


def identityTransform(X, y): return X, y


labelMethods = [
    # ("questionnaire", utils.nhanesToQuestionnaireSet),
    ("hypertension", utils.labelHypertensionBasedOnPaper)
    # ("lab_thresh", utils.labelCVRBasedOnLabMetrics(2)),
    # ("cardiovascular_codebook", utils.labelCVRBasedOnCardiovascularCodebook),
    # ("cvr_death", utils.nhanesToMortalitySet),
    # ("cvr_death_extra", utils.labelCVrBasedOnNHANESMortalityAndExtraFactors)
]


splits = 10
fold = model_selection.StratifiedKFold(
    n_splits=splits, shuffle=True, random_state=randomState)
target = 'f1'
testSize = .2

replacements = [
    lambda: impute.SimpleImputer(strategy='mean')
]

selections = [
    # lambda: feature_selection.SelectPercentile(),
    lambda: preprocessing.FunctionTransformer(lambda x: x)
    # lambda: CorrelationSelection(threshold=0.01)
]

outliers = [
    lambda: FunctionSampler(func=iqrBinaryClassesRemoval)
    # lambda: FunctionSampler(func=lambda x, y: (x, y))
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

for n, f in labelMethods:
    X, Y = f(dataset)
    X.describe().to_csv(f"results/{n}_dataset_info.csv")
    Y.value_counts(normalize=True).to_csv(f"results/{n}_label_info.csv")

# print("No Sampling")

# cvModels = ml.generatePipelines(
#     models, scalers, replacements, selections)  # type: ignore

# for nl in labelMethods:
#     ml.labelThenTrainTest(nl, cvModels, scoringConfig, target,  # type: ignore
#                           testSize, fold, dataset, "results/no_sampling")

# -------

# print("SMOTETOMEK")

# cvModels = ml.generatePipelinesWithSampling(
#     models, scalers, replacements, [lambda: combine.SMOTETomek()], selections)  # type: ignore

# for nl in labelMethods:
#     ml.labelThenTrainTest(nl, cvModels, scoringConfig, target,  # type: ignore
#                           testSize, fold, dataset, "results/smotetomek")

# -------

# print("SMOTEENN")

# cvModels = ml.generatePipelinesWithSampling(
#     models, scalers, replacements, [lambda: combine.SMOTEENN()], selections)  # type: ignore

# for nl in labelMethods:
#     ml.labelThenTrainTest(nl, cvModels, scoringConfig, target,  # type: ignore
#                           testSize, fold, dataset, "results/smoteenn")

# -------

print("KMEANS UNDERSAMPLING")

cvModels = ml.generatePipelinesWithSamplingAndOutlier(
    models, scalers, replacements, [
        lambda: FunctionSampler(func=trans.kMeansUnderSampling)],
    selections, outliers)

for nl in labelMethods:
    ml.labelThenTrainTest(nl, cvModels, scoringConfig, target,  # type: ignore
                          testSize, fold, dataset, "results/kmeans_undersampling")


# print("imblearn KMEANS UNDERSAMPLING")

# cvModels = ml.generatePipelinesWithSampling(
#     models, scalers, replacements, [lambda: under_sampling.ClusterCentroids()], selections)

# for nl in labelMethods:
#     ml.labelThenTrainTest(nl, cvModels, scoringConfig, target,  # type: ignore
#                           testSize, fold, dataset, "results/imblearn_kmeans_undersampling")
