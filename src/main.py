from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import ensemble, model_selection, neighbors, neural_network, preprocessing, svm, linear_model
from BalancedKFold import BalancedKFold, RepeatedBalancedKFold
import utils
import nhanse_dl
from ml_pipeline import run_ml_pipeline
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.impute import SimpleImputer


# CONFIGURATION VARIABLES
scores = {"precision": make_scorer(precision_score, average="binary", zero_division=0),
          "recall": make_scorer(recall_score, average="binary", zero_division=0),
          "f1": make_scorer(f1_score, average="binary", zero_division=0),
          "accuracy": make_scorer(accuracy_score)}
scoring = scores.keys()
scoring_types = ["train", "test"]
scoring_res = [f"mean_{x}_{y}" for x in scoring_types for y in scoring]
csv_columns = ["model"] + scoring_res
SAVE_DIR = "../results"
FIT_SCORE = "precision"
max_iter = 200

random_state = 42
folds = 10
fold_repeats = 10
folding_strats = [
    model_selection.KFold(n_splits=folds, shuffle=True,
                          random_state=random_state),
    model_selection.StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=random_state),
    BalancedKFold(n_splits=folds, shuffle=True, random_state=random_state),
    # RepeatedBalancedKFold(n_splits=folds)
]


models = [
    (linear_model.LogisticRegression(random_state=random_state, max_iter=max_iter),
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

    (linear_model.SGDClassifier(shuffle=True, random_state=random_state),
        {"loss": ["perceptron", "log", "perceptron"], "penalty":["l1", "l2"]}),

    (linear_model.RidgeClassifier(random_state=random_state), [
        {"solver": [
            "sag", "svd", "lsqr", "cholesky", "sparse_cg", "sag", "saga"]},
        {"solver": ["lbfgs"], "positive": [True]}
    ]),

    (ensemble.RandomForestClassifier(random_state=random_state), {
     "class_weight": [None, "balanced", "balanced_subsample"]}
     ),

    (neighbors.KNeighborsClassifier(), {"weights": ["uniform", "distance"]}),

    (neural_network.MLPClassifier(shuffle=True, max_iter=max_iter), {
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


normalizers = [
    None,
    preprocessing.MinMaxScaler(),
    preprocessing.Normalizer(),
    preprocessing.StandardScaler(),
    preprocessing.RobustScaler()
]

NHANSE_DATA_FILES = [
    nhanse_dl.NHANSERequest(
        (1999, 2000),
        ["LAB13.XPT", "LAB13AM.XPT", "LAB10AM.XPT", "LAB18.XPT", "CDQ.XPT", "DIQ.XPT", "BPQ.XPT",
            "BMX.XPT", "DEMO.XPT", "BPX.XPT"]),
    nhanse_dl.NHANSERequest(
        (2001, 2002),
        ["L13_B.XPT", "L13AM_B.XPT", "L10AM_B.XPT", "L10_2_B.XPT", "CDQ_B.XPT", "DIQ_B.XPT", "BPQ_B.XPT",
            "BMX_B.XPT", "DEMO_B.XPT", "BPX_B.XPT"]),
    nhanse_dl.NHANSERequest(
        (2003, 2004),
        ["L13_C.XPT", "L13AM_C.XPT", "L10AM_C.XPT", "CDQ_C.XPT", "DIQ_C.XPT", "BPQ_C.XPT", "BMX_C.XPT",
            "DEMO_C.XPT", "BPX_C.XPT"]),
    nhanse_dl.NHANSERequest(
        (2005, 2006),
        ["TCHOL_D.XPT", "TRIGLY_D.XPT", "HDL_D.XPT", "GLU_D.XPT", "CDQ_D.XPT", "DIQ_D.XPT", "BPQ_D.XPT",
            "BMX_D.XPT", "DEMO_D.XPT", "BPX_D.XPT"]),
    nhanse_dl.NHANSERequest(
        (2007, 2008),
        ["TCHOL_E.XPT", "TRIGLY_E.XPT", "HDL_E.XPT", "GLU_E.XPT", "CDQ_E.XPT", "DIQ_E.XPT", "BPQ_E.XPT",
            "BMX_E.XPT", "DEMO_E.XPT", "BPX_E.XPT"]),
    nhanse_dl.NHANSERequest(
        (2009, 2010),
        ["TCHOL_F.XPT", "TRIGLY_F.XPT", "HDL_F.XPT", "GLU_F.XPT", "CDQ_F.XPT", "DIQ_F.XPT", "BPQ_F.XPT",
            "BMX_F.XPT", "DEMO_F.XPT", "BPX_F.XPT"]),
    nhanse_dl.NHANSERequest(
        (2011, 2012),
        ["TCHOL_G.XPT", "TRIGLY_G.XPT", "HDL_G.XPT", "GLU_G.XPT", "CDQ_G.XPT", "DIQ_G.XPT", "BPQ_G.XPT",
            "BMX_G.XPT", "DEMO_G.XPT", "BPX_G.XPT"]),
    nhanse_dl.NHANSERequest(
        (2013, 2014),
        ["TCHOL_H.XPT", "TRIGLY_H.XPT", "HDL_H.XPT", "GLU_H.XPT", "CDQ_H.XPT", "DIQ_H.XPT", "BPQ_H.XPT",
            "BMX_H.XPT", "DEMO_H.XPT", "BPX_H.XPT"])
]


# NOTE: NHANSE dataset early on had different variables names for some features
# CombineFeatures is used to combine these features into a single feature

def meanReplacement(x): return utils.map_dataframe(
    lambda y: SimpleImputer().fit_transform(y), x)


def answeredYesOnQuestion(x: pd.DataFrame): return x.applymap(
    lambda d: 1 if d == 1 else 0)


experimentConfigs = [
    utils.Experiment("lab_work", [
        utils.CombineFeatures.rename(
            "LBXTC", "Total_Chol", postProcess=meanReplacement),
        utils.CombineFeatures.rename(
            "LBDLDL", "LDL", postProcess=meanReplacement),
        utils.CombineFeatures(
            ["LBDLDL", "LBXHDD", "LBDHDD"], "HDL", postProcess=meanReplacement),
        utils.CombineFeatures(
            ["LBXSGL", "LB2GLU", "LBXGLU"], "FBG", postProcess=meanReplacement),
        utils.CombineFeatures.rename(
            "LBXTR", "FBG", postProcess=meanReplacement),
    ]),
    utils.Experiment("classic_heart_attack", [
        utils.CombineFeatures.rename(
            "DIQ010", "DIABETES", False, answeredYesOnQuestion),
        utils.CombineFeatures.rename(
            "BPQ020", "HYPERTEN", False, answeredYesOnQuestion),
        # utils.CombineFeatures.rename(
        #     "CDQ001", "CHEST_PAIN", False, answeredYesOnQuestion),
    ]),
    utils.Experiment("measurements", [
        utils.CombineFeatures.rename(
            "BMXBMI", "BMI", postProcess=meanReplacement),
        utils.CombineFeatures.rename(
            "BMXWAIST", "WC", postProcess=meanReplacement),
        utils.CombineFeatures(
            ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"], "SYSTOLIC", postProcess=meanReplacement),  # Might be better to take average
        utils.CombineFeatures(
            ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"], "DIASTOLIC", postProcess=meanReplacement),
        utils.CombineFeatures.rename(
            "RIAGENDR", "GENDER", answeredYesOnQuestion),
        utils.CombineFeatures.rename("RIDAGEYR", "AGE"),
    ])
]


NHANSE_DATASET = nhanse_dl.get_nhanse_mortality_dataset(
    NHANSE_DATA_FILES)  # Load once

LINKED_DATASET = NHANSE_DATASET.loc[NHANSE_DATASET.ELIGSTAT == 1, :]
DEAD_DATASET = LINKED_DATASET.loc[LINKED_DATASET.MORTSTAT == 1, :]

DEAD_DATASET.describe().to_csv("../results/dead_dataset_info.csv")

experimentConfigs = [utils.combineExperiments(
    "all_features", experimentConfigs)]


for run_name, combine_directions in experimentConfigs:
    print(f"Experiment: {run_name}")
    utils.ensure_directory_exists(f"{SAVE_DIR}/{run_name}")

    res = run_ml_pipeline(folding_strats, DEAD_DATASET, combine_directions, scores, models,
                          normalizers, csv_columns, scoring_res, SAVE_DIR, run_name, fit_score=FIT_SCORE)
