from nhanes_cvr.types import XYPair
import nhanes_cvr.utils as utils
import seaborn as sns
import nhanes_cvr.mlProcess as ml
from nhanes_cvr.config import testSize, scoringConfig, allPipelines, target, fold
from sklearn import datasets
import pandas as pd

# Matplotlib/Seaborn Theming
sns.set_theme(style='darkgrid', palette='pastel')

dataset = utils.get_nhanes_dataset()
dataset.dtypes.to_csv("results/feature_types.csv")

# TODO Include train score for refit in test_results (Or some other file)


def getIris(_) -> XYPair:
    X, Y = datasets.load_iris(return_X_y=True)
    Y = (Y == 1)
    return (pd.DataFrame(X), pd.Series(Y))


labelMethods = [
    ("iris", getIris),

    # ("hypertension", utils.labelHypertensionBasedOnPaper),
    # ("lab_thresh", utils.labelCVRBasedOnLabMetrics(2)),
    # ("cardiovascular_codebook", utils.labelCVRBasedOnCardiovascularCodeBook),
    # ("questionnaire", utils.labelQuestionnaireSet),
    # ("cvr_death", utils.nhanesToMortalitySet),
    # ("cvr_death_extra", utils.labelCVrBasedOnNHANESMortalityAndExtraFactors)
]

for n, f in labelMethods:
    X, Y = f(dataset)
    utils.makeDirectoryIfNotExists(f"results/{n}")
    X.describe().to_csv(f"results/{n}/dataset_info.csv")
    Y.value_counts(normalize=True).to_csv(f"results/{n}/label_info.csv")
    X.dtypes.to_csv(f"results/{n}/dataset_types.csv")


(allTrain, allTest) = ml.runAllLabellers(labelMethods, allPipelines, scoringConfig, target,
                                         testSize, fold, dataset, "results")

allTrain.to_csv("results/all_train_results.csv")
allTest.to_csv("results/all_test_results.csv")

print(allTrain)
