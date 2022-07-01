from datetime import datetime
from sklearn import model_selection
import nhanes_cvr.utils as utils
import seaborn as sns
from nhanes_cvr.config import dataset, gridSearchSelections, testSize, scoringConfig, models, scalers, randomState
import nhanes_cvr.mlProcess as ml

# Matplotlib/Seaborn Theming
sns.set_theme(style='darkgrid', palette='pastel')


labelMethods = [
    # ("questionnaire", utils.nhanesToQuestionnaireSet),
    ("cvr_death", utils.nhanesToMortalitySet),
    # ("lab_thresh", utils.labelCVRBasedOnLabMetrics(2)),
    # ("cardiovascular_codebook", utils.labelCVRBasedOnCardiovascularCodebook)
]


cvModels = ml.generatePipelines(models, scalers)
splits = 10
fold = model_selection.StratifiedKFold(
    n_splits=splits, shuffle=True, random_state=randomState)
target = 'f1'
testSize = .2


for nl in labelMethods:
    ml.labelThenTrainUsingMultipleSelectors(
        nl, dataset, gridSearchSelections, cvModels, scoringConfig, target, testSize, fold, "results")  # type: ignore
