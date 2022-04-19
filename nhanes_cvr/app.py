import nhanes_cvr.combinefeatures as cf
from sklearn import ensemble, model_selection, neighbors, neural_network, preprocessing, svm, linear_model
from nhanes_cvr.BalancedKFold import BalancedKFold, RepeatedBalancedKFold
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from nhanes_dl import download, types
import nhanes_cvr.gridsearch as gs
import nhanes_cvr.utils as utils


# CONFIGURATION VARIABLES
scores = {"precision": make_scorer(precision_score, average="binary", zero_division=0),
          "recall": make_scorer(recall_score, average="binary", zero_division=0),
          "f1": make_scorer(f1_score, average="binary", zero_division=0),
          "accuracy": make_scorer(accuracy_score)}
scoring = scores.keys()
targetScore = "precision"
maxIter = 200
randomState = 42
folds = 10
foldRepeats = 10

foldingStrategies = [
    model_selection.KFold(n_splits=folds, shuffle=True,
                          random_state=randomState),
    model_selection.StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=randomState),
    # BalancedKFold(n_splits=folds, shuffle=True, random_state=random_state),
    # RepeatedBalancedKFold(n_splits=folds)
]


models = [
    (linear_model.LogisticRegression(random_state=randomState, max_iter=maxIter),
     [
        {
            "C": [.5, 1],
            "penalty": ["l2"],
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
        },
        {
            "C": [.5, 1],
            "penalty": ["l1"],
            "solver": ["liblinear", "saga"]
        }
    ]),

    (linear_model.SGDClassifier(shuffle=True, random_state=randomState),
        {"loss": ["perceptron", "log", "perceptron"], "penalty":["l1", "l2"]}),

    (linear_model.RidgeClassifier(random_state=randomState), [
        {"solver": [
            "sag", "svd", "lsqr", "cholesky", "sparse_cg", "sag", "saga"]},
        {"solver": ["lbfgs"], "positive": [True]}
    ]),

    (ensemble.RandomForestClassifier(random_state=randomState), {
     "class_weight": [None, "balanced", "balanced_subsample"]}
     ),

    (neighbors.KNeighborsClassifier(), {"weights": ["uniform", "distance"]}),

    (neural_network.MLPClassifier(shuffle=True, max_iter=maxIter), {
     "activation": ["logistic", "tanh", "relu"],
     "solver": ["lbfgs", "sgd", "adam"],
     "learning_rate":["invscaling"]
     }),

    (svm.LinearSVC(random_state=42), [
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
    # Everything past this point has the same feature (Mostly)
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


def replaceMissingWithNo(X):
    return cf.replaceMissingWith(1, X)


# # NOTE: NHANSE dataset early on had different variables names for some features
# # CombineFeatures is used to combine these features into a single feature
combine_configs = [
    # - Lab Work -
    cf.rename("LBXTC", "Total_Chol", postProcess=cf.meanMissingReplacement),
    cf.rename("LBDLDL", "LDL", postProcess=cf.meanMissingReplacement),
    cf.create(["LBDHDL", "LBXHDD", "LBDHDD"], "HDL",
              postProcess=cf.meanMissingReplacement),
    # utils.CombineFeatures.rename(
    #     "LBDHDD", "HDL", postProcess=utils.meanReplacement),
    cf.create(["LBXSGL", "LB2GLU", "LBXGLU"], "FBG",
              postProcess=cf.meanMissingReplacement),
    # utils.CombineFeatures.rename(
    #     "LBXGLU", "FBG", postProcess=utils.meanReplacement),  # glucose
    # triglercyides
    cf.rename("LBXTR", "TG", postProcess=cf.meanMissingReplacement),
    # - Questionaire -
    cf.rename(
        "DIQ010", "DIABETES", replaceMissingWithNo),
    cf.rename(
        "BPQ020", "HYPERTEN", replaceMissingWithNo),
    cf.rename(
        "CDQ001", "CHEST_PAIN", replaceMissingWithNo),
    # - Measurements -
    cf.rename("BMXBMI", "BMI", postProcess=cf.meanMissingReplacement),
    cf.rename("BMXWAIST", "WC", postProcess=cf.meanMissingReplacement),
    cf.create(
        ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"], "SYSTOLIC", combineStrategy=cf.meanCombine, postProcess=cf.meanMissingReplacement),
    cf.create(["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"],
              "DIASTOLIC", combineStrategy=cf.meanCombine, postProcess=cf.meanMissingReplacement),
    cf.rename("RIAGENDR", "GENDER", postProcess=replaceMissingWithNo),
    cf.rename("RIDAGEYR", "AGE"),
]


NHANES_DATASET = utils.cache_nhanes("./data/nhanes.csv",
                                    lambda: download.downloadCodebooksWithMortalityForYears(downloadConfig))
LINKED_DATASET = NHANES_DATASET.loc[NHANES_DATASET.ELIGSTAT == 1, :]
DEAD_DATASET = LINKED_DATASET.loc[LINKED_DATASET.MORTSTAT == 1, :]
withoutScalingFeatures = ["DIABETES", "HYPERTEN", "GENDER"]

print(f"Entire Dataset: {NHANES_DATASET.shape}")
print(f"Linked Mortality Dataset: {LINKED_DATASET.shape}")
print(f"Dead Dataset: {DEAD_DATASET.shape}")

DEAD_DATASET.describe().to_csv("./results/dead_dataset_info.csv")

Y = utils.labelCauseOfDeathAsCVR(DEAD_DATASET)
X = cf.runCombines(combine_configs, DEAD_DATASET)

scalingConfigs = gs.createScalerConfigsIgnoreFeatures(
    scalers, X, withoutScalingFeatures)
gridSearchConfigs = gs.createGridSearchConfigs(
    models, scalingConfigs, foldingStrategies, [scores])

res = gs.runMultipleGridSearchAsync(gridSearchConfigs, targetScore, X, Y)
resultsDF = gs.resultsToDataFrame(res)
gs.plotResults3d(resultsDF, targetScore)
resultsDF.to_csv("./results/results.csv")
