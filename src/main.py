from distutils.file_util import write_file
import pandas as pd
from sklearn import ensemble, model_selection, svm
from sklearn.model_selection import train_test_split
import Utils
from sklearn import linear_model

# TODO
# [x] Need to convert mortstat data to correct data types INT
# [x] Figure out how to Combining NHANSE data
# [x] Check what the 6 lab measurements were and find there nhanse files
# [x] Enter all NHANSE Years with correct data files
# [] Validate if Dataset is correctly put together
# [x] Setup X,Y from dataset
# [] Setup 60,30,10 split for training,val, and testing
# [] Setup graphs of training results

# Forget about LDLC for now
markers = ["TotalChol", "LDL", "HDL", "FBG", "TG", "LDLC"]


NHANSE_DATA_FILES = [
    ((2005, 2006), ["TCHOL_D.XPT", "TRIGLY_D.XPT", "HDL_D.xpt", "GLU_D.xpt"]),
    ((2007, 2008), ["TCHOL_E.XPT", "TRIGLY_E.XPT", "HDL_E.xpt", "GLU_E.xpt"]),
    ((2009, 2010), ["TCHOL_F.XPT", "TRIGLY_F.XPT", "HDL_F.xpt", "GLU_F.xpt"]),
    ((2011, 2012), ["TCHOL_G.XPT", "TRIGLY_G.XPT", "HDL_G.xpt", "GLU_G.xpt"])]

# IS Fasting glucose different then glucose
# All measurements in (mg/dl)
features_descritions = [("LBXTC", "Total Chol"), ("LBDLDL", "LDL Chol"),
                        ("LBDHDD", "HDL Chol"), ("LBXGLU", "Fasting Glucose"), ("LBXTR", "Triglyceride")]
features = [x for x, _ in features_descritions]

dataset = Utils.get_nhanse_mortality_dataset(NHANSE_DATA_FILES)

# print(dataset.columns)
X = dataset.loc[:, features].assign(
    CVR=dataset.UCOD_LEADING.apply(Utils.labelCauseOfDeathAsCVR)).dropna()
Y = X.CVR
X = X.loc[:, features]

print(f"Dataset Size: {X.shape}")
print(f"True Sample Count: {Y.sum()}")
print(f"True Sample Percentage: {Y.sum() / X.shape[0]}%")


def train_model(model, X, y, n_folds, scoring):
    cv = model_selection.StratifiedKFold(
        n_folds, shuffle=True, random_state=42)
    res = model_selection.cross_validate(
        model, X, y, cv=cv, scoring=scoring, return_train_score=True)

    return model, res


def transformToCSVData(modelRes):
    m, res = modelRes
    scores = [Utils.avg(res[x]) for x in scoring_res]
    return [m.__class__] + scores


scoring = ["accuracy", "f1", "precision", "recall"]
scoring_types = ["train", "test"]
scoring_res = [f"{x}_{y}" for x in scoring_types for y in scoring]
csvColumns = ["model"] + scoring_res
folds = [2, 4, 5]
models = [
    linear_model.LogisticRegression(),
    ensemble.RandomForestClassifier(),
    svm.LinearSVC()]

for k in folds:
    modelRes = [train_model(m, X, Y, k, scoring) for m in models]
    csvData = [transformToCSVData(x) for x in modelRes]

    csvDataframe = pd.DataFrame(
        csvData, columns=csvColumns)

    csvDataframe.to_csv(f"{k}fold_model_results.csv")
