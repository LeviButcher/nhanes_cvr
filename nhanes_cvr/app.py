from datetime import datetime
import pandas as pd
from sklearn import ensemble, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
import nhanes_cvr.gridsearch as gs
import nhanes_cvr.utils as utils
import seaborn as sns
from nhanes_cvr.config import dataset, gridSearchSelections, testSize, randomState, scoringConfig, models, targetScore, scalers
import nhanes_cvr.rewrite as rw

# Matplotlib/Seaborn Theming
sns.set_theme()

# Setup functions to label cause of death differently
normalCVRDeath = [utils.LeadingCauseOfDeath.HEART_DISEASE,
                  utils.LeadingCauseOfDeath.CEREBROVASCULAR_DISEASE]
diabetesDeath = [utils.LeadingCauseOfDeath.DIABETES_MELLITUS]
expandedCVRDeath = normalCVRDeath + diabetesDeath
labelMethods = [
    # ("questionnaire", utils.nhanesToQuestionnaireSet),
    ("cvr_death", utils.nhanesToMortalitySet),
    # ("cvr_death_within_time", utils.nhanesToMortalityWithinTimeSet(10))
    # ("diabetes_death", utils.labelCVR(diabetesDeath)),
    # ("cvr_diabetes_death", utils.labelCVR(expandedCVRDeath))
]


cvModels = rw.generatePipelines(models, scalers)
splits = 10
fold = model_selection.StratifiedKFold(n_splits=splits)
target = 'f1'
testSize = .2

start = datetime.now()
for labelName, getY in labelMethods:
    originalX, originalY = getY(dataset)
    saveDir = f"results/{labelName}"

    utils.makeDirectoryIfNotExists(saveDir)

    print(dataset.shape)

    for name, selectF in gridSearchSelections:
        print(name)
        runSaveDir = f"{saveDir}/{name}"
        utils.makeDirectoryIfNotExists(runSaveDir)

        X, Y = selectF((originalX, originalY))

        rw.trainTestProcess(cvModels, scoringConfig, targetScore,
                            testSize, fold, X, Y, runSaveDir)
