from sklearn import ensemble, model_selection, neighbors, neural_network, decomposition, preprocessing, svm, linear_model
from BalancedKFold import BalancedKFold, RepeatedBalancedKFold
import utils
import nhanse_dl
from ml_pipeline import run_ml_pipeline


# CONFIGURATION VARIABLES
scoring = ["accuracy", "f1", "precision", "recall"]
scoring_types = ["train", "test"]
scoring_res = [f"mean_{x}_{y}" for x in scoring_types for y in scoring]
csv_columns = ["model"] + scoring_res
SAVE_DIR = "../results"

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
    (linear_model.LogisticRegression(multi_class="ovr",
                                     random_state=random_state),
     {"penalty": ['l1', 'l2'], "C":[0, .5, 1],
      "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
      }),
    (linear_model.SGDClassifier(shuffle=True, random_state=random_state),
        {"loss": ["perceptron", "log", "perceptron"], "penalty":["l1", "l2"]}),
    (linear_model.RidgeClassifier(random_state=random_state), {"solver": [
     "sag", "svd", "lsqr", "cholesky", "sparse_cg", "sag", "saga", "ldfgs"]}),
    (ensemble.RandomForestClassifier(random_state=42), {
     "class_weight": ["balanced", "balanced_subsample"]}),
    (neighbors.KNeighborsClassifier(), {"weights": ["uniform", "distance"]}),
    (neural_network.MLPClassifier(shuffle=True), {
     "activation": ["logistic", "tanh", "relu"],
     "solver": ["lbfgs", "sgd", "adam"],
     "learning_rate":["invscaling"]
     }),
    (svm.LinearSVC(random_state=42), {"penalty": ['l1', 'l2'],
                                      "C": [.05, 1],
                                      }),
    (svm.OneClassSVM(), {
        "kernel": ["linear", "poly", 'rbf', 'sigmoid']
    })
]


normalizers = [
    None,
    preprocessing.MinMaxScaler(),
    preprocessing.Normalizer(),
    preprocessing.StandardScaler()
]

NHANSE_DATA_FILES = [
    nhanse_dl.NHANSERequest(
        (1999, 2000), ["LAB13.XPT", "LAB13AM.XPT", "LAB10AM.XPT", "LAB18.XPT"]),
    nhanse_dl.NHANSERequest(
        (2001, 2002), ["L13_B.XPT", "L13AM_B.XPT", "L10AM_B.XPT", "L10_2_B.XPT"]),
    nhanse_dl.NHANSERequest(
        (2003, 2004), ["L13_C.XPT", "L13AM_C.XPT", "L10AM_C.XPT"]),
    nhanse_dl.NHANSERequest(
        (2005, 2006), ["TCHOL_D.XPT", "TRIGLY_D.XPT", "HDL_D.xpt", "GLU_D.xpt"]),

    nhanse_dl.NHANSERequest(
        (2007, 2008), ["TCHOL_E.XPT", "TRIGLY_E.XPT", "HDL_E.xpt", "GLU_E.xpt"]),
    nhanse_dl.NHANSERequest(
        (2009, 2010), ["TCHOL_F.XPT", "TRIGLY_F.XPT", "HDL_F.xpt", "GLU_F.xpt"])
]

# IS Fasting glucose different then glucose
# All measurements in (mg/dl)
# features_descritions = [("LBXTC", "Total Chol"), ("LBDLDL", "LDL Chol"),
#                         ("LBDHDD", "HDL Chol"), ("LBXGLU", "Fasting Glucose"), ("LBXTR", "Triglyceride")]

# Is there a difference in HDL for nhanse data?
# Fasting Glucose vs Glucose

# | Year  | Total_Chol | LDL_Chol | HDL_Chol | Glucose (FBG) | Triglyceride |
# | ----  | ---------  | -------- | -------- | ------------- | ------------ |
# | 99-00 |      LBXTC | LBDLDL   | LBDHDL   | LBXSGL        | LBXTR        |
# | 01-02 |      LBXTC | LBDLDL   | LBDHDL   | LB2GLU        | LBXTR        |
# | 03-04 |      LBXTC | LBDLDL   | LBXHDD   | LBXGLU        | LBXTR        |
# | 05-06 |      LBXTC | LBDLDL   | LBDHDD   | LBXGLU        | LBXTR        |
# | 07-08 |      LBXTC | LBDLDL   | LBDHDD   | LBXGLU        | LBXTR        |
# | 09-10 |      LBXTC | LBDLDL   | LBDHDD   | LBXGLU        | LBXTR        |

# NOTE: NHANSE dataset early on had different variables names for some features
# combine directions is used to combine these features into a single feature
combine_directions = [
    (["LBXTC"], "Total_Chol"),  # VALID MAP
    (["LBDLDL"], "LDL"),
    (["LBDHDL", "LBXHDD", "LBDHDD"], "HDL"),
    (["LBXSGL", "LB2GLU", "LBXGLU"], "FBG"),
    (["LBXTR"], "TG"),
    (["UCOD_LEADING"], "UCOD_LEADING")]


# DATASET LOADING
features = [x for _, x in combine_directions]
nhanse_dataset = nhanse_dl.get_nhanse_mortality_dataset(NHANSE_DATA_FILES)

print(f"True Diabetes: {(nhanse_dataset.DIABETES == 1).sum()}")
print(f"True HyperTen: {(nhanse_dataset.HYPERTEN == 1).sum()}")

print(nhanse_dataset.UCOD_LEADING.value_counts())

dataset = utils.combine_df_columns(combine_directions, nhanse_dataset)

# DATASET TRANSFORMATION
# TODO: Improve this tranformation to be sure that the Y is dropped from X
X = dataset.loc[:, features].assign(
    CVR=dataset.UCOD_LEADING.apply(utils.labelCauseOfDeathAsCVR)).dropna()
Y = X.CVR
X = X.loc[:, features].drop(columns=['UCOD_LEADING'])

print(f"Dataset Size: {X.shape}")
print(f"True Sample Count: {Y.sum()}")
print(f"True Sample Percentage: {Y.sum() / X.shape[0] * 100}%")

run_name = "experiment1"

# RUN EXPERIMENT1
res = run_ml_pipeline(folding_strats, X, Y, scoring, models,
                      normalizers, csv_columns, scoring_res, SAVE_DIR, run_name, fit_score="f1")
