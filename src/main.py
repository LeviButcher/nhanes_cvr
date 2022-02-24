import pandas as pd
from sklearn import ensemble, model_selection, neighbors, neural_network, svm
import utils
from sklearn import linear_model
import nhanse_dl
import matplotlib.pyplot as plt


# TODO
# [x] Need to convert mortstat data to correct data types INT
# [x] Figure out how to Combining NHANSE data
# [x] Check what the 6 lab measurements were and find there nhanse files
# [x] Enter all NHANSE Years with correct data files
# [x] Validate if Dataset is correctly put together
# [x] Setup X,Y from dataset
# [x] Pull in old nhanse data (Requires mapping two columns together)
# [x] map model class name down to just it actual class name for csv data
# [x] Add in chart for quick visualizations of methods
# [x] Figure out what models to use in ml pipeline (Need more anomaly detection)
# [] Add caching of nhanse data files
# [] Perform pca then plot PC1 vs PC2 and highlight samples that are positive for CVR
# [] Check the HDL and Glucose features used are actually the same


def train_model(model, X, y, n_folds, scoring):
    cv = model_selection.StratifiedKFold(
        n_folds, shuffle=True, random_state=42)
    res = model_selection.cross_validate(
        model, X, y, cv=cv, scoring=scoring, return_train_score=True)

    return model, res


def transform_to_csv_data(modelRes):
    m, res = modelRes
    scores = [utils.avg(res[x]) for x in scoring_res]
    return [m.__class__.__name__] + scores


def run_ml_pipeline(folds, X, Y, scoring, models):
    for k in folds:
        print(f"Running {k}fold")
        modelRes = [train_model(m, X, Y, k, scoring) for m in models]
        csv_data = [transform_to_csv_data(x) for x in modelRes]

        csv_dataframe = pd.DataFrame(
            csv_data, columns=csv_columns)

        x = csv_dataframe.model
        y = csv_dataframe.test_f1
        plt.bar(x, y)
        plt.xticks(x, rotation=-15, fontsize="x-small")
        plt.xlabel("Model")
        plt.ylabel("Test F1 Score")
        plt.savefig(f"../data/{k}fold_model_plot_results.png")
        plt.close()

        csv_dataframe.to_csv(f"../data/{k}fold_model_results.csv")


# CONFIGURATION VARIABLES
scoring = ["accuracy", "f1", "precision", "recall"]
scoring_types = ["train", "test"]
scoring_res = [f"{x}_{y}" for x in scoring_types for y in scoring]
csv_columns = ["model"] + scoring_res
folds = [2, 4, 5]
models = [
    linear_model.LogisticRegression(),
    linear_model.RidgeClassifier(),
    ensemble.RandomForestClassifier(),
    neighbors.KNeighborsClassifier(),
    neural_network.MLPClassifier()
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
dataset = utils.combine_df_columns(combine_directions, nhanse_dataset)


X = dataset.loc[:, features].assign(
    CVR=dataset.UCOD_LEADING.apply(utils.labelCauseOfDeathAsCVR)).dropna()
Y = X.CVR
X = X.loc[:, features]

X = (X - X.min()) / (X.max() - X.min())  # Mean Normalization

print(f"Dataset Size: {X.shape}")
print(f"True Sample Count: {Y.sum()}")
print(f"True Sample Percentage: {Y.sum() / X.shape[0]}%")

# RUN PIPELINE
run_ml_pipeline(folds, X, Y, scoring, models)
