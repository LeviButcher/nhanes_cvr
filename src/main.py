from sklearn import ensemble, model_selection, neighbors, neural_network, decomposition, preprocessing, svm, linear_model
from BalancedKFold import BalancedKFold, RepeatedBalancedKFold
import utils
import nhanse_dl
from ml_pipeline import run_ml_pipeline
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
# [x] Perform pca then plot PC1 vs PC2 and highlight samples that are positive for CVR
# [] Add caching of nhanse data files
# [] Add GridSearch to ml_pipeline
# [] Check the HDL and Glucose features used are actually the same
# [] Add threaded processes to ml_pipeline


# CONFIGURATION VARIABLES
scoring = ["accuracy", "f1", "precision", "recall"]
scoring_types = ["train", "test"]
scoring_res = [f"{x}_{y}" for x in scoring_types for y in scoring]
csv_columns = ["model"] + scoring_res

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
    linear_model.LogisticRegression(
        solver="lbfgs", max_iter=100, penalty='l2', random_state=random_state),
    linear_model.SGDClassifier(
        loss='perceptron', penalty='l1', shuffle=True, random_state=True),
    linear_model.RidgeClassifier(solver='sag'),
    ensemble.RandomForestClassifier(),
    neighbors.KNeighborsClassifier(),
    neural_network.MLPClassifier(),
    svm.LinearSVC(),
    svm.OneClassSVM()
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
X = dataset.loc[:, features].assign(
    CVR=dataset.UCOD_LEADING.apply(utils.labelCauseOfDeathAsCVR)).dropna()
Y = X.CVR
X = X.loc[:, features]

print(f"Dataset Size: {X.shape}")
print(f"True Sample Count: {Y.sum()}")
print(f"True Sample Percentage: {Y.sum() / X.shape[0] * 100}%")

# RUN PIPELINE
res = run_ml_pipeline(folding_strats, X, Y, scoring, models,
                      normalizers, csv_columns, scoring_res)

res.to_csv('../results/all_results.csv')

# Group Fold Results into Plot
for foldName, data in res.groupby(['foldingStrat']):
    plt.title(foldName)
    for normName, data2 in data.groupby(['normalizer']):
        f1 = data2.test_f1
        models = data2.model
        norms = data2.normalizer
        # encodedNorms = preprocessing.LabelEncoder().fit_transform(norms)

        plt.scatter(models, f1, label=normName)
    plt.xticks(models, rotation=-15, fontsize="x-small")
    plt.xlabel("Models")
    plt.ylabel("F1")
    plt.legend(loc="best")
    plt.savefig(f"../results/{foldName}_plot.png")
    plt.close()

# Display All Results in 3d plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
normEncoder = preprocessing.LabelEncoder()
modelEncoder = preprocessing.LabelEncoder()
res = res.assign(normalizer_enc=normEncoder.fit_transform(res.normalizer),
                 model_enc=modelEncoder.fit_transform(res.model))

for foldName, data in res.groupby(['foldingStrat']):
    f1 = data.test_f1
    models = data.model_enc
    norms = data.normalizer_enc

    ax.scatter(models, norms, f1, label=foldName)

plt.xticks(res.model_enc, modelEncoder.inverse_transform(res.model_enc))
plt.yticks(res.normalizer_enc,
           normEncoder.inverse_transform(res.normalizer_enc))
plt.xlabel("Models")
plt.ylabel("Normalizations")
plt.zlabel("F1")
plt.legend(loc="best")
plt.savefig(f"../results/all_results_3d_plot.png")
plt.close()
