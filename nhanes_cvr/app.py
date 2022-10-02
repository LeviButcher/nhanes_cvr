from matplotlib import pyplot as plt
from sklearn import model_selection, preprocessing, impute
from sklearn.decomposition import PCA
import nhanes_cvr.utils as utils
import seaborn as sns
import nhanes_cvr.mlProcess as ml
from imblearn import pipeline, FunctionSampler, under_sampling
from nhanes_cvr.transformers import DropTransformer, bestScoreByClosestToMean, bestScoreByClosestToMedian, highestScoreIndex, lowestScoreIndex
from nhanes_cvr.config import testSize, scoringConfig, models, scalers, randomState
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
import pandas as pd

# Matplotlib/Seaborn Theming
sns.set_theme(style='darkgrid', palette='pastel')

dataset = utils.get_nhanes_dataset()
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
    # lambda: feature_selection.SelectPercentile(),
    lambda: preprocessing.FunctionTransformer()
    # lambda: CorrelationSelection(threshold=0.01)
]

outliers = [
    # lambda: FunctionSampler(func=iqrBinaryClassesRemoval)
    # Use when Doing iqrBinaryClassesRemoval on all dataset
    lambda: preprocessing.FunctionTransformer()
]

kValues = [2, 3, 4]

clusterMethods = [KMeans, KMedoids]

bestScoresFunctions = [highestScoreIndex, lowestScoreIndex, bestScoreByClosestToMean,
                       bestScoreByClosestToMedian]

# keep = (dataset.dtypes == 'float64')
# dataset = dataset.loc[:, keep]

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

    *utils.generateKMeansUnderSampling(kValues, clusterMethods, bestScoresFunctions)
    # ("cluster_centroids", under_sampling.ClusterCentroids)
]

labellerResults = []

for n, sampler in samplerRuns:
    print(n)
    cvModels = ml.generatePipelinesWithSampling(
        models, scalers, replacements, [sampler], selections)  # type: ignore

    res = [ml.labelThenTrainTest(nl, cvModels, scoringConfig, target, testSize, fold, dataset, f"results/{n}")
           .assign(labeller=n)
           for nl in labelMethods]

    labellerResults.append(pd.concat(res))

columns = ["labeller", "modelAppr", "scalingAppr",
           "selectionAppr", "accuracy", "precision", "recall", "f1", "auc_roc"]

allCSV = pd.concat(labellerResults, ignore_index=True).loc[:, columns]
allCSV.to_csv("results/all_results.csv")
allCSV.to_html("results/all_results.html")

bestResults = allCSV.groupby(by="labeller")["f1"].idxmax()

bestCSV = allCSV.loc[bestResults, :]
bestCSV.to_csv("results/best_results.csv")
bestCSV.to_html("results/best_results.html")

bestByModelResults = allCSV.groupby(
    by=["labeller", "modelAppr"])["f1"].idxmax()

bestModelCSV = allCSV.loc[bestByModelResults, :]
bestModelCSV.to_csv("results/best_model_results.csv")
bestModelCSV.to_html("results/best_model_results.html")
