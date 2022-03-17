import numpy as np
import pandas as pd
from sklearn import ensemble, model_selection, neighbors, neural_network, preprocessing, svm, linear_model
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from BalancedKFold import BalancedKFold, RepeatedBalancedKFold
from ml_pipeline import run_ml_pipeline


# CONFIGURATION VARIABLES
scores = {"precision": make_scorer(precision_score, average="weighted", zero_division=0),
          "recall": make_scorer(recall_score, average="weighted", zero_division=0),
          "f1": make_scorer(f1_score, average="weighted", zero_division=0),
          "accuracy": make_scorer(accuracy_score)}
scoring = scores.keys()
scoring_types = ["train", "test"]
scoring_res = [f"mean_{x}_{y}" for x in scoring_types for y in scoring]
csv_columns = ["model"] + scoring_res
SAVE_DIR = "../pipeline_test"
FIT_SCORE = "precision"

random_state = 42
folds = 10
fold_repeats = 10
folding_strats = [
    model_selection.StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=random_state),
    # BalancedKFold(n_splits=folds, shuffle=True, random_state=random_state),
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
    preprocessing.StandardScaler()
]


mnist = pd.read_csv('../data/mnist_train.csv').sample(n=500)
Y = mnist.iloc[:, 0]
X = mnist.iloc[:, 1:]
run_name = "mnist"

res = run_ml_pipeline(folding_strats, X, Y, scores, models,
                      normalizers, csv_columns, scoring_res, SAVE_DIR, run_name, fit_score=FIT_SCORE)
