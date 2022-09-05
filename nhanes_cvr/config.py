import toolz as toolz
import nhanes_cvr.combinefeatures as cf
from sklearn import ensemble, neighbors, neural_network, preprocessing, svm, linear_model, cluster
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score, roc_auc_score
from nhanes_dl import download
import nhanes_cvr.utils as utils
import nhanes_dl.types as types
import numpy as np

# CONFIGURATION VARIABLES
scoringConfig = {"precision": make_scorer(precision_score, average="binary", zero_division=0),
                 "recall": make_scorer(recall_score, average="binary", zero_division=0),
                 "f1": make_scorer(f1_score, average="binary", zero_division=0),
                 "accuracy": make_scorer(accuracy_score)
                 }
targetScore = "f1"
maxIter = 200
randomState = np.random.RandomState(0)
folds = 10
foldRepeats = 10
testSize = .20
correlationThreshold = 0.05
zScoreThreshold = 2.9
nullThreshold = 3

models = [
    # (linear_model.LogisticRegression,
    #  [
    #      {
    #          'model__solver': ['sag'],
    #          'model__penalty': ['l2'],
    #          'model__class_weight': [None, 'balanced'],
    #          #  'model__C': stats.expon(scale=100)
    #      },
    #      {
    #          'model__solver': ['saga'],
    #          'model__penalty': ['elasticnet', 'l2', 'l1'],
    #          'model__class_weight': [None, 'balanced'],
    #          'model__random_state': [randomState],
    #          #  'model__C': stats.expon(scale=100)
    #      }
    #  ]
    #  ),
    ((ensemble.RandomForestClassifier),
     {
        # 'model__n_estimators': np.arange(100, 500, 5),
        # 'model__min_samples_split': np.arange(1, 30, 2),
        # 'model__max_depth': np.arange(20, 100, 10),
        # 'model__min_samples_leaf': np.arange(1, 50, 20),
        # 'model__class_weight': [None, 'balanced', 'balanced_subsample'],
        # 'model__criterion': ['gini', 'entropy', 'log_loss'],
        # 'model__max_features': np.arange(5, 200, 5),
        'model__random_state': [randomState]
    }),
    # (neural_network.MLPClassifier, {}
    #  #  [{
    #  #      'model__solver': ['adam'],
    #  #      'model__activation': ['relu', 'tanh', 'logistic'],
    #  #      'model__random_state': [randomState]
    #  #  }, {
    #  #      # 'model__solver': ['sgd'],
    #  #      # 'model__activation': ['relu', 'tanh', 'logistic'],
    #  #      'model__random_state': [randomState],
    #  #      # 'model__learning_rate': ['adaptive', 'invscaling']
    #  #  }]
    #  ),
    (svm.SVC, {
        # 'model__C': stats.expon(scale=100),
        # 'model__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        # 'model__class_weight': [None, 'balanced'],
    }),
    # (neighbors.KNeighborsClassifier, {}
    #  {
    #      "model__weights": ["uniform", "distance"],
    #      "model__n_neighbors": np.arange(1, 20, 10),
    #      "model__leaf_size": np.arange(30, 100, 10)
    #  }
    #  ),
]


models = [(m, {}) for m, _ in models]

scalers = [
    preprocessing.FunctionTransformer,
    preprocessing.MinMaxScaler,
    # preprocessing.Normalizer,
    preprocessing.StandardScaler,
    # preprocessing.RobustScaler
]


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

    cf.rename("MCQ010", "TOLD_HAVE_ASTHMA", postProcess=standardYesNoProcessor)
]


# Save CSV of combinationConfigs
ccDF = cf.combineFeaturesToDataFrame(combineConfigs)
ccDF.to_csv("./results/handpicked_features.csv")
