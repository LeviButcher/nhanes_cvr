from typing import List
import pandas as pd
import nhanes_cvr.combinefeatures as cf
from sklearn import ensemble, model_selection, neighbors, neural_network, preprocessing, svm, linear_model
from nhanes_cvr.BalancedKFold import BalancedKFold, RepeatedBalancedKFold
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from nhanes_dl import download
import nhanes_cvr.gridsearch as gs
import nhanes_cvr.utils as utils
import nhanes_dl.types as types


# CONFIGURATION VARIABLES
scoringConfig = {"precision": make_scorer(precision_score, average="binary", zero_division=0),
                 "recall": make_scorer(recall_score, average="binary", zero_division=0),
                 "f1": make_scorer(f1_score, average="binary", zero_division=0),
                 "accuracy": make_scorer(accuracy_score)}
targetScore = "precision"
maxIter = 200
randomState = 42
folds = 10
foldRepeats = 10
testSize = .20

foldingStrategies = [
    model_selection.KFold(n_splits=folds, shuffle=True,
                          random_state=randomState),
    model_selection.StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=randomState),
    BalancedKFold(n_splits=folds, shuffle=True, random_state=randomState),
    # model_selection.RepeatedKFold(n_splits=10, n_repeats=10),
    # model_selection.RepeatedStratifiedKFold(n_splits=10, n_repeats=10),
    # RepeatedBalancedKFold(n_splits=10, n_repeats=10)
]


# Store Model in Thunks to ensure recreation of new model every GridSearch
models = [
    (lambda: linear_model.LogisticRegression(random_state=randomState, max_iter=maxIter),
     [
        {
            "C": [.2, .4, .6, .8, 1],
            "penalty": ["l2", "none"],
            "solver": ["newton-cg", "liblinear", "sag"],
            "class_weight": [None, "balanced"]
        },
        {
            "C": [.2, .4, .6, .8, 1],
            "penalty": ["l1", "none"],
            "solver": ["liblinear", "saga"],
            "class_weight": [None, "balanced"]
        },
        {
            "C": [.2, .4, .6, .8, 1],
            "penalty": ["l2", "l1", "none"],
            "solver": ["saga"],
            "class_weight": [None, "balanced"],
            "l1_ratio": [.5]
        },
        {
            "C": [.2, .4, .6, .8, 1],
            "penalty": ["elasticnet"],
            "solver": ["saga"],
            "class_weight": [None, "balanced"],
            "l1_ratio": [.3, .5, .8]
        }
    ]),

    (lambda: linear_model.SGDClassifier(shuffle=True, random_state=randomState),
        {"loss": ["perceptron", "log_loss", "perceptron"], "penalty":["l1", "l2"]}),

    (lambda: linear_model.RidgeClassifier(random_state=randomState), [
        {"solver": [
            "sag", "svd", "lsqr", "cholesky", "sparse_cg", "sag", "saga"]},
        {"solver": ["lbfgs"], "positive": [True]}
    ]),

    (lambda: ensemble.RandomForestClassifier(random_state=randomState), {
     "class_weight": [None, "balanced", "balanced_subsample"],
     "max_features": ["sqrt", "log2"],  # May be best exploring this variable
     "criterion": ["gini", "entropy", "log_loss"]
     }),

    (lambda: neighbors.KNeighborsClassifier(),
     {"weights": ["uniform", "distance"],
      "n_neighbors": [5, 10],
      "leaf_size": [30, 50]
      }),

    (lambda: neural_network.MLPClassifier(shuffle=True, max_iter=maxIter), {
     "activation": ["logistic", "tanh", "relu"],
     "solver": ["sgd", "adam"],
     "learning_rate":["invscaling", "adaptive"]
     }),

    (lambda: svm.LinearSVC(random_state=42), [
        {
            "loss": ["hinge"],
            "penalty": ['l2'],
            "C": [.05, 1],
        }, {
            "loss": ["squared_hinge"],
            "penalty": ['l2'],
            "C": [.05, 1],
        }
    ])
]


scalers = [
    None,
    preprocessing.MinMaxScaler(),
    preprocessing.Normalizer(),
    preprocessing.StandardScaler(),
    preprocessing.RobustScaler()
]

# NOTE: Most studies I've seen don't use the earlier years of nhanes
# Probably cause it's harder to combine with them
downloadConfig = {
    download.CodebookDownload(types.ContinuousNHANES.First,
                              "LAB13", "LAB13AM", "LAB10AM", "LAB18", "CDQ",
                              "DIQ", "BPQ", "BMX", "DEMO", "BPX"),
    download.CodebookDownload(types.ContinuousNHANES.Second,
                              "L13_B", "L13AM_B", "L10AM_B", "L10_2_B",
                              "CDQ_B", "DIQ_B", "BPQ_B", "BMX_B", "DEMO_B", "BPX_B"),
    download.CodebookDownload(types.ContinuousNHANES.Third,
                              "L13_C", "L13AM_C", "L10AM_C", "CDQ_C", "DIQ_C",
                              "BPQ_C", "BMX_C", "DEMO_C", "BPX_C"),
    # Everything past this point has the same codebooks
    download.CodebookDownload(types.ContinuousNHANES.Fourth,
                              "TCHOL_D", "TRIGLY_D", "HDL_D", "GLU_D", "CDQ_D",
                              "DIQ_D", "BPQ_D", "BMX_D", "DEMO_D", "BPX_D"),
    download.CodebookDownload(types.ContinuousNHANES.Fifth,
                              "TCHOL_E", "TRIGLY_E", "HDL_E", "GLU_E", "CDQ_E",
                              "DIQ_E", "BPQ_E", "BMX_E", "DEMO_E", "BPX_E"),
    download.CodebookDownload(types.ContinuousNHANES.Sixth,
                              "TCHOL_F", "TRIGLY_F", "HDL_F", "GLU_F", "CDQ_F",
                              "DIQ_F", "BPQ_F", "BMX_F", "DEMO_F", "BPX_F"),
    download.CodebookDownload(types.ContinuousNHANES.Seventh,
                              "TCHOL_G", "TRIGLY_G", "HDL_G", "GLU_G", "CDQ_G",
                              "DIQ_G", "BPQ_G", "BMX_G", "DEMO_G", "BPX_G"),
    download.CodebookDownload(types.ContinuousNHANES.Eighth,
                              "TCHOL_H", "TRIGLY_H", "HDL_H", "GLU_H", "CDQ_H",
                              "DIQ_H", "BPQ_H", "BMX_H", "DEMO_H", "BPX_H"),
}

# Only generates for years Forth-Eighth
downloadConfig = utils.generateDownloadConfig(["TCHOL", "TRIGLY", "HDL",
                                               "GLU", "CDQ", "DIQ",
                                               "BPQ", "BMX", "DEMO",
                                               "BPX", "SMQ", "DBQ",
                                               "PAQ", "CBC", "GHB",
                                               "BIOPRO", "UIO"])


def standardYesNoProcessor(X):
    # value of 1 is YES
    # Set any other value to 0
    return X.apply(lambda x: 1 if x == 1 else 0)


def highestValueNullReplacer(X):
    m = X.max()
    return X.fillna(m)

# NOTE: NHANSE dataset early on had different variables names for some features
# CombineFeatures is used to combine these features into a single feature


# Should set up postProcess to drop rows if postProcess returns a smaller series
# Might be able to change this to specify type feature should be
combineConfigs = [
    cf.rename("RIDAGEYR", "AGE"),
    cf.rename("RIAGENDR", "GENDER"),
    cf.rename("RIDRETH1", "Race"),
    cf.rename("LBXTC", "Total_Chol", postProcess=cf.meanMissingReplacement),
    cf.rename("LBDLDL", "LDL", postProcess=cf.meanMissingReplacement),

    cf.rename("LBDHDD", "HDL", postProcess=cf.meanMissingReplacement),
    cf.rename("LBXGLU", "FBG", postProcess=cf.meanMissingReplacement),
    # cf.create(["LBDHDL", "LBXHDD", "LBDHDD"], "HDL",
    #           postProcess=cf.meanMissingReplacement),
    # cf.create(["LBXSGL", "LB2GLU", "LBXGLU"], "FBG",
    #           postProcess=cf.meanMissingReplacement),

    # triglercyides
    cf.rename("LBXTR", "TG", postProcess=cf.meanMissingReplacement),
    cf.rename("DIQ010", "DOCTOR_TOLD_HAVE_DIABETES",
              postProcess=standardYesNoProcessor),
    cf.rename("DIQ160", "TOLD_HAVE_PREDIABETES",
              postProcess=standardYesNoProcessor),
    cf.rename("DIQ170", "TOLD_AT_RISK_OF_DIABETES",
              postProcess=standardYesNoProcessor),
    cf.rename("DIQ200A", "CONTROLLING_WEIGHT",
              postProcess=standardYesNoProcessor),

    cf.rename("DIQ050", "NOW_TAKING_INSULIN",
              postProcess=standardYesNoProcessor),
    cf.rename("BPQ020", "HIGH_BLOOD_PRESSURE",
              postProcess=standardYesNoProcessor),
    cf.rename("BPQ030", "HIGH_BLOOD_PRESSURE_TWO_OR_MORE",
              postProcess=standardYesNoProcessor),
    cf.rename("BPQ052", "TOLD_HAVE_PREHYPERTENSION",
              postProcess=standardYesNoProcessor),
    cf.rename("BPQ057", "TOLD_BORDERLINE_HYPERTENSION",
              postProcess=standardYesNoProcessor),
    cf.rename("BPQ050A", "TAKEN_DRUGS_FOR_HYPERTEN",
              postProcess=standardYesNoProcessor),
    cf.rename("BPQ080", "TOLD_HAVE_HIGH_CHOL",
              postProcess=standardYesNoProcessor),
    cf.rename("BPQ090A", "TOLD_EAT_LESS_FAT_FOR_CHOL",
              postProcess=standardYesNoProcessor),
    cf.rename("BPQ090B", "TOLD_REDUCE_WEIGHT_FOR_CHOL",
              postProcess=standardYesNoProcessor),
    cf.rename("BPQ090C", "TOLD_EXERCISE_FOR_CHOL",
              postProcess=standardYesNoProcessor),
    cf.rename("BPQ090D", "TOLD_PRESCRIPTION_FOR_CHOL",
              postProcess=standardYesNoProcessor),
    cf.rename("BPQ100A", "NOW_EATING_LESS_FAT_FOR_CHOL",
              postProcess=standardYesNoProcessor),
    cf.rename("BPQ100B", "NOW_CONTROLLING_WEIGHT_FOR_CHOL",
              postProcess=standardYesNoProcessor),
    cf.rename("BPQ100C", "NOW_INCREASING_EXERCISE_FOR_CHOL",
              postProcess=standardYesNoProcessor),
    cf.rename("BPQ100D", "NOW_TAKING_PRESCRIPTION_FOR_CHOL",
              postProcess=standardYesNoProcessor),
    cf.rename(
        "CDQ001", "CHEST_PAIN", postProcess=standardYesNoProcessor),
    cf.rename("BMXBMI", "BMI", postProcess=cf.meanMissingReplacement),
    cf.rename("BMXWAIST", "WC", postProcess=cf.meanMissingReplacement),
    cf.rename("BMXHT", "HEIGHT", postProcess=cf.meanMissingReplacement),
    cf.rename("BMXWT", "WEIGHT", postProcess=cf.meanMissingReplacement),
    cf.rename("BMXARMC", "ARM_CIRCUMFERENCE",
              postProcess=cf.meanMissingReplacement),
    cf.create(["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"], "SYSTOLIC",
              combineStrategy=cf.meanCombine, postProcess=cf.meanMissingReplacement),
    cf.create(["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"], "DIASTOLIC",
              combineStrategy=cf.meanCombine, postProcess=cf.meanMissingReplacement),
    cf.rename("SMQ020", "SMOKED_AT_LEAST_100_IN_LIFE",
              postProcess=standardYesNoProcessor),
    cf.rename("SMQ040", "CURRENTLY_SMOKES",
                        postProcess=standardYesNoProcessor),
    cf.rename("DBQ700", "HOW_HEALTHY_IS_DIET",
              postProcess=highestValueNullReplacer),
    # Seems that this isn't in every year
    # cf.rename("PAQ560", "TIMES_PER_WEEK_EXERCISE_HARD",
    #           postProcess=highestValueNullReplacer),
    # cf.rename("PAD590", "HOURS_WATCHED_TV",
    #           postProcess=highestValueNullReplacer),
    # cf.rename("PAD600", "HOURS_COMPUTER_USE",
    #           postProcess=highestValueNullReplacer),
    # cf.rename("PAQ605", "DOES_WORK_INVOLVE_VIGOROUS_ACTIVITY",
    #           postProcess=standardYesNoProcessor),
    cf.rename("LBXHGB", "HEMOGOBLIN", postProcess=cf.meanMissingReplacement),
    cf.rename("LBXGH", "GLYCOHEMOGLOBIN",
              postProcess=cf.meanMissingReplacement),
    cf.rename("LBXSBU", "BLOOD_UREA_NITROGEN",
              postProcess=cf.meanMissingReplacement),
    cf.rename("LBXSCR", "CREATINE",
              postProcess=cf.meanMissingReplacement),
]

notNullCombineConfig = [
    # Demographics
    cf.rename("RIDAGEYR", "AGE"),
    cf.rename("RIAGENDR", "GENDER"),
    cf.rename("RIDRETH1", "Race"),

    # Lab
    cf.rename("LBXTC", "Total_Chol", postProcess=cf.meanMissingReplacement),
    cf.rename("LBDLDL", "LDL", postProcess=cf.meanMissingReplacement),
    cf.rename("LBXTR", "TG", postProcess=cf.meanMissingReplacement),
    cf.rename("LBDHDD", "HDL", postProcess=cf.meanMissingReplacement),

    cf.create(["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"], "SYSTOLIC",
              combineStrategy=cf.meanCombine, postProcess=cf.meanMissingReplacement),
    cf.create(["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"], "DIASTOLIC",
              combineStrategy=cf.meanCombine, postProcess=cf.meanMissingReplacement),

    cf.rename("LBXGLU", "FBG", postProcess=cf.meanMissingReplacement),
    cf.rename("LBXIN", "INSULIN", postProcess=cf.meanMissingReplacement),

    cf.rename("LBXHGB", "HEMOGOBLIN", postProcess=cf.meanMissingReplacement),
    cf.rename("LBXGH", "GLYCOHEMOGLOBIN",
              postProcess=cf.meanMissingReplacement),

    # Very few have this
    # cf.rename("LBXAPB", "APOLIPOPROTEIN",
    #           postProcess=cf.meanMissingReplacement),


    # # Takes away like 700 samples
    # cf.rename("URXUIO", "IODINE",
    #           postProcess=cf.meanMissingReplacement),
    # cf.rename("URXUCR", "CREATINE",
    #           postProcess=cf.meanMissingReplacement),




    # # Questionaire
    cf.rename("CDQ001", "CHEST_PAIN", postProcess=standardYesNoProcessor),
    cf.rename("CDQ010", "SHORTNESS_OF_BREATHS",
              postProcess=standardYesNoProcessor),
    # # # Could add more from CDQ
    cf.rename("SMQ020", "SMOKED_AT_LEAST_100_IN_LIFE",
              postProcess=standardYesNoProcessor),

    # # Might add Sleep, Weight History,
]

# Download NHANES
updateCache = utils.doesNHANESNeedRedownloaded(downloadConfig)
NHANES_DATASET = utils.cache_nhanes("./data/nhanes.csv",
                                    lambda: download.downloadCodebooksWithMortalityForYears(downloadConfig), updateCache=updateCache)

# Process NHANES
LINKED_DATASET = NHANES_DATASET.loc[NHANES_DATASET.ELIGSTAT == 1, :]
DEAD_DATASET = LINKED_DATASET.loc[LINKED_DATASET.MORTSTAT == 1, :]
ALIVE_DATASET = LINKED_DATASET.loc[LINKED_DATASET.MORTSTAT == 0, :]

print(f"Entire Dataset: {NHANES_DATASET.shape}")
print(f"Linked Mortality Dataset: {LINKED_DATASET.shape}")
print(f"Dead Dataset: {DEAD_DATASET.shape}")
print(f"Alive Dataset: {ALIVE_DATASET.shape}")
DEAD_DATASET.describe().to_csv("./results/dead_dataset_info.csv")

dataset = DEAD_DATASET  # Quickly allows running on other datasets

# TODO: Expose API for this in nhanes-dl
mortalityCols = [x for x in download.mortality_colnames
                 if x not in download.drop_columns]
featuresToScale = [cf.meanMissingReplacement.__name__]


# --- HAND PICKED FEATURES TRAINING ---


def runHandPickedFeatures():
    runName = "handpicked"
    ccDF = cf.combineFeaturesToDataFrame(combineConfigs)
    ccDF.to_csv("./results/handpicked_features.csv")

    X = cf.runCombines(combineConfigs, dataset)
    Y = utils.labelCauseOfDeathAsCVR(dataset)

    withoutScalingFeatures = [
        c.combinedName for c in combineConfigs if c.postProcess.__name__ not in featuresToScale]
    print(f"Not Scaling: {withoutScalingFeatures}")

    scalingConfigs = gs.createScalerConfigsIgnoreFeatures(
        scalers, X, withoutScalingFeatures)
    runGridSearch(X, Y, scalingConfigs, runName)


def runHandPickedNoNulls():
    runName = "handPickedNoNulls"
    ccDF = cf.combineFeaturesToDataFrame(notNullCombineConfig)
    ccDF.to_csv("./results/handpickedNoNull_features.csv")
    keepNullConfig = cf.noPostProcessingForAll(notNullCombineConfig)
    X = cf.runCombines(keepNullConfig, dataset)
    Y = utils.labelCauseOfDeathAsCVR(dataset)
    notNull = X.notnull().all(axis=1)
    X = X.loc[notNull, :]
    Y = Y.loc[notNull]
    print(X.shape)
    print(Y.shape)
    print(f"True % = {Y[Y == 1].count() / (Y.shape[0])}")

    # exit()

    withoutScalingFeatures = [
        c.combinedName for c in notNullCombineConfig if c.postProcess.__name__ not in featuresToScale]
    print(f"Not Scaling: {withoutScalingFeatures}")

    scalingConfigs = gs.createScalerConfigsIgnoreFeatures(
        scalers, X, withoutScalingFeatures)
    runGridSearch(X, Y, scalingConfigs, runName)


def runHandPickedNoNullsAndOutliers():
    runName = "handPickedNoNullsAndOutliers"
    ccDF = cf.combineFeaturesToDataFrame(notNullCombineConfig)
    ccDF.to_csv("./results/handpickedNoNull_features.csv")
    keepNullConfig = cf.noPostProcessingForAll(notNullCombineConfig)
    X = cf.runCombines(keepNullConfig, dataset)
    Y = utils.labelCauseOfDeathAsCVR(dataset)
    notNull = X.notnull().all(axis=1)
    X = X.loc[notNull, :]
    Y = Y.loc[notNull]

    noOutliers = utils.remove_outliers(2.9, X)
    X = X.loc[noOutliers, :]
    Y = Y.loc[noOutliers]

    print(X.shape)
    print(Y.shape)
    print(f"True % = {Y[Y == 1].count() / (Y.shape[0])}")

    withoutScalingFeatures = [
        c.combinedName for c in notNullCombineConfig if c.postProcess.__name__ not in featuresToScale]
    print(f"Not Scaling: {withoutScalingFeatures}")
    # exit()

    scalingConfigs = gs.createScalerConfigsIgnoreFeatures(
        scalers, X, withoutScalingFeatures)
    runGridSearch(X, Y, scalingConfigs, runName)


# --- CORRELATION TRAINING ---

def runCorrelationFeatureSelectionDropNulls():
    runName = "correlationNoNulls"
    threshold = 0.1

    # NOTE: Should null be dropped before or after corrrelation?
    filteredDataset = dataset.assign(
        Y=utils.labelCauseOfDeathAsCVR(dataset)).drop(columns=mortalityCols)  # type: ignore
    cor = filteredDataset.corr()
    cor_target = abs(cor["Y"])[:-1]  # Cut off Y correlation with -1
    relevant_features = cor_target[cor_target > threshold]

    relevant_features.to_csv(f"./results/{runName}_correlatedFeatures.csv")

    X = filteredDataset.loc[:, relevant_features.index.values]
    X = X.assign(Y=filteredDataset.Y).dropna(thresh=3, axis=0)

    Y = X.Y
    X = X.drop(columns=["Y"])
    X = X.fillna(X.mean())

    scalingConfigs = gs.createScalerAllFeatures(scalers, X)
    runGridSearch(X, Y, scalingConfigs, runName)


def runCorrelationFeatureSelection():
    runName = "correlation"
    threshold = 0.1

    # Correlation Selection
    # Followed article: https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
    filteredDataset = dataset.assign(
        Y=utils.labelCauseOfDeathAsCVR(dataset)).drop(columns=mortalityCols)  # type: ignore
    cor = filteredDataset.corr()
    cor_target = abs(cor["Y"])[:-1]  # Cut off Y correlation with -1
    relevant_features = cor_target[cor_target > threshold]

    relevant_features.to_csv(f"./results/{runName}_correlatedFeatures.csv")

    X = filteredDataset.loc[:, relevant_features.index.values]
    X = X.fillna(X.mean())
    Y = filteredDataset.Y

    scalingConfigs = gs.createScalerAllFeatures(scalers, X)
    runGridSearch(X, Y, scalingConfigs, runName)


# Abstract function to quickly run the gridSearchs using different X/Y/scalingConfigs


def runGridSearch(X: pd.DataFrame, Y: pd.Series, scalingConfigs: List[gs.ScalerConfig], runName: str):
    print(f"Starting {runName}")
    print(f"X: {X.shape}")
    print(f"Y: {Y.shape}")

    trainX, testX, trainY, testY = model_selection.train_test_split(
        X, Y, test_size=testSize, random_state=randomState, stratify=Y)

    # Create Configs
    gridSearchConfigs = gs.createGridSearchConfigs(
        models, scalingConfigs, foldingStrategies, [scoringConfig])

    # Run Training
    res = gs.runMultipleGridSearchs(
        gridSearchConfigs, targetScore, trainX, trainY)
    resultsDF = gs.resultsToDataFrame(res)

    # Save Plots
    [gs.plotResultsGroupedByModel(
        resultsDF, s, f"./results/{runName}_train_groupedModelPlots") for s in gs.resultsScores(res[0][1])]

    gs.plotResults3d(resultsDF, f"mean_test_{targetScore}",
                     f"./results/{runName}_train_plotResults3d")
    resultsDF.to_csv(f"./results/{runName}_train_results.csv")

    # Output Best Info
    print("\n--- FINISHED ---\n")
    print(f"Ran {len(res)} Configs")
    gs.printBestResult(res)

    testResultsDF = gs.evaluateModels(testX, testY, scoringConfig, res)
    [gs.plotResultsGroupedByModel(
        testResultsDF, s, f"./results/{runName}_test_groupedModelPlots") for s in scoringConfig.keys()]
    gs.plotResults3d(testResultsDF, targetScore,
                     f"./results/{runName}_test_plotResults3d")
    testResultsDF.to_csv(f"./results/{runName}_test_results.csv")
