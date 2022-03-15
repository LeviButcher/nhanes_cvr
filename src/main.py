import numpy as np
from sklearn import ensemble, model_selection, neighbors, neural_network, preprocessing, svm, linear_model
from BalancedKFold import BalancedKFold, RepeatedBalancedKFold
import utils
import nhanse_dl
from ml_pipeline import run_ml_pipeline
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score


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

random_state = 42
folds = 10
fold_repeats = 10
folding_strats = [
    model_selection.StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=random_state),
    BalancedKFold(n_splits=folds, shuffle=True, random_state=random_state),
    # RepeatedBalancedKFold(n_splits=folds)
]


models = [
    (linear_model.LogisticRegression(random_state=random_state, max_iter=100),
     [
        {
            "C": np.linspace(0.1, 1, 10),
            "penalty": ["l2"],
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
        },
        {
            "C": np.linspace(0.1, 1, 10),
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

    (neural_network.MLPClassifier(shuffle=True), {
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
        ["LAB13.XPT", "LAB13AM.XPT", "LAB10AM.XPT", "LAB18.XPT", "CDQ.XPT", "DIQ.XPT", "BPQ.XPT"]),
    nhanse_dl.NHANSERequest(
        (2001, 2002),
        ["L13_B.XPT", "L13AM_B.XPT", "L10AM_B.XPT", "L10_2_B.XPT", "CDQ_B.XPT", "DIQ_B.XPT", "BPQ_B.XPT"]),
    nhanse_dl.NHANSERequest(
        (2003, 2004),
        ["L13_C.XPT", "L13AM_C.XPT", "L10AM_C.XPT", "CDQ_C.XPT", "DIQ_C.XPT", "BPQ_C.XPT"]),
    nhanse_dl.NHANSERequest(
        (2005, 2006),
        ["TCHOL_D.XPT", "TRIGLY_D.XPT", "HDL_D.XPT", "GLU_D.XPT", "CDQ_D.XPT", "DIQ_D.XPT", "BPQ_D.XPT"]),
    nhanse_dl.NHANSERequest(
        (2007, 2008),
        ["TCHOL_E.XPT", "TRIGLY_E.XPT", "HDL_E.XPT", "GLU_E.XPT", "CDQ_E.XPT", "DIQ_E.XPT", "BPQ_E.XPT"]),
    nhanse_dl.NHANSERequest(
        (2009, 2010),
        ["TCHOL_F.XPT", "TRIGLY_F.XPT", "HDL_F.XPT", "GLU_F.XPT", "CDQ_F.XPT", "DIQ_F.XPT", "BPQ_F.XPT"])
]


# NOTE: NHANSE dataset early on had different variables names for some features
# combine directions is used to combine these features into a single feature
# TODO Make into its own class, with custom functions, Could easily make use of monoids here
EXPERIMENT_CONFIG = [
    ("lab_work", [
        (["LBXTC"], "Total_Chol"),
        (["LBDLDL"], "LDL"),
        (["LBDHDL", "LBXHDD", "LBDHDD"], "HDL"),
        (["LBXSGL", "LB2GLU", "LBXGLU"], "FBG"),
        (["LBXTR"], "TG"),
        (["UCOD_LEADING"], "UCOD_LEADING")]),
    ("classic_heart_attack",
     [
         (["DIQ010"], "DIABETES"),
         (["BPQ020"], "HYPERTEN"),
         #  (["DIABETES"], "DIABETES"),
         #  (["HYPERTEN"], "HYPERTEN"),
         (["CDQ001"], "CHEST_PAIN"),
         #  (["CDQ005"], "STANDING_RELIEVE"),
         #  (["CDQ009G"], "PAIN_LEFT_ARM"),
         #  (["CDQ008"], "SEVERE_CHEST_PAIN"),
         (["UCOD_LEADING"], "UCOD_LEADING")])
]


NHANSE_DATASET = nhanse_dl.get_nhanse_mortality_dataset(
    NHANSE_DATA_FILES)  # Load once


def combine_configs(experiment_config):
    variables = utils.unique(
        [x for _, config in experiment_config for x in config])
    return ("combine_all", variables)


def get_dataset(combine_directions):
    features = [x for _, x in combine_directions]
    dataset = utils.combine_df_columns(combine_directions, NHANSE_DATASET)

    print(f"Dataset Size: {dataset.shape}")
    X = dataset.loc[:, features].assign(
        CVR=dataset.UCOD_LEADING.apply(utils.labelCauseOfDeathAsCVR)
    ).drop(columns=['UCOD_LEADING'])

    print(X.isna().sum())
    X = X.dropna()

    Y = X.CVR
    X = X.drop(columns=["CVR"])

    print(f"Dataset Size (After dropping NAN): {X.shape}")
    print(f"True Sample Count: {Y.sum()}")
    print(f"True Sample Percentage: {Y.sum() / X.shape[0] * 100}%")

    return X, Y


EXPERIMENT_CONFIG.append(combine_configs(EXPERIMENT_CONFIG))


# print(NHANSE_DATASET.describe())

for run_name, combine_directions in EXPERIMENT_CONFIG:
    print(f"Experiment: {run_name}")
    X, Y = get_dataset(combine_directions)
    X.describe().to_csv(f"{SAVE_DIR}/{run_name}_feature_description.csv")

    res = run_ml_pipeline(folding_strats, X, Y, scores, models,
                          normalizers, csv_columns, scoring_res, SAVE_DIR, run_name, fit_score=FIT_SCORE)
