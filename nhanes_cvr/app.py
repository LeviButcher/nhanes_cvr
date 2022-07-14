from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
import nhanes_cvr.utils as utils
import seaborn as sns
from nhanes_cvr.config import dataset, gridSearchSelections, testSize, scoringConfig, models, scalers, randomState
import nhanes_cvr.mlProcess as ml
from imblearn import under_sampling, combine

# Matplotlib/Seaborn Theming
sns.set_theme(style='darkgrid', palette='pastel')


labelMethods = [
    ("questionnaire", utils.nhanesToQuestionnaireSet),
    ("lab_thresh", utils.labelCVRBasedOnLabMetrics(2)),
    ("cardiovascular_codebook", utils.labelCVRBasedOnCardiovascularCodebook),
    ("cvr_death", utils.nhanesToMortalitySet),
    # ("cvr_death_extra", utils.labelCVrBasedOnNHANESMortalityAndExtraFactors)
]

samplers = [
    lambda: combine.SMOTEENN(random_state=randomState)
]

cvModels = ml.generateSamplingPipelines(
    samplers, models, scalers)  # type: ignore
splits = 10
fold = model_selection.StratifiedKFold(
    n_splits=splits, shuffle=True, random_state=randomState)
target = 'f1'
testSize = .2


# TODO Current Issue - Need to drop columns in the codebooks used for classification to make
# sure any info that could easily help identify Y isn't given
# For Lab classification mmol measurements need to be dropped.

# Question
# How should the dataset be split into train/test???

# cvModels = ml.generatePipelines(models, scalers)  # type: ignore

# # for nl in labelMethods:
# #     ml.labelThenTrainUsingMultipleSelectors(
# #         nl, dataset, gridSearchSelections, cvModels, scoringConfig, target, testSize, fold, "results/no_sampling")


cvModels = ml.generateSamplingPipelines(
    samplers, models, scalers)  # type: ignore

for nl in labelMethods:
    ml.labelThenTrainUsingMultipleSelectors(
        nl, dataset, gridSearchSelections, cvModels, scoringConfig, target, testSize, fold, "results/sampling")  # type: ignore

# ml.mortalityAnalysis(
#     dataset, gridSearchSelections[0][1], labelMethods, testSize)
