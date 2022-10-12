from sklearn import model_selection, preprocessing, impute
import nhanes_cvr.utils as utils
import seaborn as sns
import nhanes_cvr.mlProcess as ml
from nhanes_cvr.config import testSize, scoringConfig, randomState, allModels

# Matplotlib/Seaborn Theming
sns.set_theme(style='darkgrid', palette='pastel')

dataset = utils.get_nhanes_dataset()
dataset.dtypes.to_csv("results/feature_types.csv")


labelMethods = [
    # ("questionnaire", utils.labelQuestionnaireSet),
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

scaling = [
    preprocessing.FunctionTransformer,
    preprocessing.MinMaxScaler,
    preprocessing.StandardScaler,
]


for n, f in labelMethods:
    X, Y = f(dataset)
    X.describe().to_csv(f"results/{n}_dataset_info.csv")
    Y.value_counts(normalize=True).to_csv(f"results/{n}_label_info.csv")


pipelines = ml.generatePipelines(allModels, scaling, replacements, selections)


(allTrain, allTest) = ml.runAllLabellers(labelMethods, pipelines, scoringConfig, target,
                                         testSize, fold, dataset, "results")

allTrain.to_csv("results/all_train_results.csv")
allTest.to_csv("results/all_test_results.csv")

print(allTrain)
