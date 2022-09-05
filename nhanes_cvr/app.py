from matplotlib import pyplot as plt
import pandas as pd
from sklearn import feature_selection, model_selection, preprocessing, impute
from sklearn.decomposition import PCA
import nhanes_cvr.utils as utils
import seaborn as sns
import nhanes_cvr.mlProcess as ml
from imblearn import pipeline, FunctionSampler, over_sampling, combine, under_sampling
from nhanes_cvr.transformers import DropTransformer, iqrBinaryClassesRemoval
import nhanes_cvr.transformers as trans
from nhanes_cvr.config import testSize, scoringConfig, models, scalers, randomState

utils.isolatedRun()
exit()


dataset = utils.get_nhanes_dataset()


# Matplotlib/Seaborn Theming
sns.set_theme(style='darkgrid', palette='pastel')

dataset.dtypes.to_csv("results/feature_types.csv")


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
    lambda: feature_selection.SelectPercentile(),
    # lambda: preprocessing.FunctionTransformer(lambda x: x)
    # lambda: CorrelationSelection(threshold=0.01)
]

outliers = [
    # lambda: FunctionSampler(func=iqrBinaryClassesRemoval)
    # Use when Doing iqrBinaryClassesRemoval on all dataset
    lambda: preprocessing.FunctionTransformer()
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
    plt.close()

for n, f in labelMethods:
    X, Y = f(dataset)
    X.describe().to_csv(f"results/{n}_dataset_info.csv")
    Y.value_counts(normalize=True).to_csv(f"results/{n}_label_info.csv")


samplerRuns = [
    ("no_sampling", lambda: FunctionSampler()),
    ("random_undersampling", under_sampling.RandomUnderSampler),
    # ("smotetomek", combine.SMOTETomek),
    # ("smote", over_sampling.SMOTE),
    # ("smoteenn", combine.SMOTEENN),
    ("kmeans_undersampling", lambda: FunctionSampler(func=trans.kMeansUnderSampling)),
    # ("cluster_centroids", under_sampling.ClusterCentroids)
]

for n, sampler in samplerRuns:
    print(n)
    cvModels = ml.generatePipelinesWithSampling(
        models, scalers, replacements, [sampler], selections)  # type: ignore
    for nl in labelMethods:
        ml.labelThenTrainTest(nl, cvModels, scoringConfig, target,  # type: ignore
                              testSize, fold, dataset, f"results/{n}")
