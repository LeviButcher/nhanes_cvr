from datetime import datetime
from locale import normalize
import pandas as pd
import nhanes_cvr.gridsearch as gs
import nhanes_cvr.utils as utils
import seaborn as sns
from nhanes_cvr.config import dataset, mortalityCols, gridSearchSelections, testSize, randomState, scoringConfig, models, foldingStrategies, targetScore

# Matplotlib/Seaborn Theming
sns.set_theme()

# Setup functions to label cause of death differently
normalCVRDeath = [utils.LeadingCauseOfDeath.HEART_DISEASE,
                  utils.LeadingCauseOfDeath.CEREBROVASCULAR_DISEASE]
diabetesDeath = [utils.LeadingCauseOfDeath.DIABETES_MELLITUS]
expandedCVRDeath = normalCVRDeath + diabetesDeath
labelMethods = [
    ("questionnaire", utils.nhanesToQuestionnaireSet),
    ("cvr_death", utils.nhanesToMortalitySet),
    # ("cvr_death_within_time", utils.nhanesToMortalityWithinTimeSet(10))
    # ("diabetes_death", utils.labelCVR(diabetesDeath)),
    # ("cvr_diabetes_death", utils.labelCVR(expandedCVRDeath))
]

start = datetime.now()
for labelName, getY in labelMethods:
    originalX, originalY = getY(dataset)
    saveDir = f"results/{labelName}"

    utils.makeDirectoryIfNotExists(saveDir)

    print(dataset.shape)

    runInfoSeries = []
    for name, selectF, getScalingConfigs in gridSearchSelections:
        print(name)
        runSaveDir = f"{saveDir}/{name}"
        utils.makeDirectoryIfNotExists(runSaveDir)

        X, Y = selectF((originalX, originalY))
        scalingConfigs = getScalingConfigs(X)
        runResults = []
        print(X.shape)
        print(Y.value_counts(normalize=True))

        for scaleConfig in scalingConfigs:
            scaledX = gs.runScaling(scaleConfig, X, Y)
            scalerName = gs.getScalerName(scaleConfig)
            print(f"SCALING: {scalerName}")

            results = gs.runAndEvaluateGridSearch(scaledX, Y, testSize,
                                                  randomState, scoringConfig, models,
                                                  foldingStrategies, targetScore, runSaveDir).assign(scaler=scalerName)  # type: ignore
            runResults.append(results)

        allResults = pd.concat(runResults)
        allResults.to_csv(f"{runSaveDir}/results.csv")

        # Train Plot
        for s in gs.getTestScoreNames(scoringConfig):
            gs.plotResultsGroupedByModel(allResults, s,  # type: ignore
                                         f"{runSaveDir}/train_groupedModelPlots",
                                         title=f"trainSet - {s}")

        # Test Plot
        for s in scoringConfig.keys():
            gs.plotResultsGroupedByModel(allResults, s,  # type: ignore
                                         f"{runSaveDir}/test_groupedModelPlots",
                                         title=f"testSet - {s}")

        totalTime = datetime.now() - start
        info = pd.Series({"name": name, "xShape": X.shape,
                          "yShape": Y.shape, "time": totalTime, "truePercent": Y[Y == 1].sum() / Y.shape[0]})
        runInfoSeries.append(info)

    runInfoDF = pd.DataFrame(runInfoSeries)
    runInfoDF.to_csv(f"{saveDir}/runInfos.csv")
