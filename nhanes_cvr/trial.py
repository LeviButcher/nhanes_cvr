from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import cluster, preprocessing, ensemble, model_selection, metrics, linear_model, neural_network
from nhanes_dl import download
import nhanes_cvr.utils as utils
import toolz as toolz
import nhanes_cvr.selection as select


dataset = pd.read_csv("./data/nhanes.csv", low_memory=False)

# Write check for duplicate SEQN


randomState = 42
testSize = .20
target = "f1"

# dataset = dataset.loc[dataset.MORTSTAT == 1, :]
keep = (dataset.RIDAGEYR >= 20) & (dataset.RHD143 != 1)
dataset = dataset.loc[keep, :]

Y = utils.labelViaQuestionnaire(dataset)
# Y = utils.labelCVR([utils.LeadingCauseOfDeath.HEART_DISEASE,
# utils.LeadingCauseOfDeath.CEREBROVASCULAR_DISEASE], dataset)
X = dataset.drop(columns=["MCQ160F", "MCQ160C",
                          "MCQ160B", "MCQ160E"]).drop(columns=download.getMortalityColumns())

dataProcessor = toolz.compose_left(
    select.dropColumns(.50),
    select.fillNullWithMean,
    select.correlationSelection(0.05)
)

print(X.shape)
print(Y.shape)  # type: ignore

(X, Y) = dataProcessor((X, Y))

X = preprocessing.StandardScaler().fit_transform(X)

print(X.shape)
print(Y.value_counts(normalize=True))


trainX, testX, trainY, testY = model_selection.train_test_split(
    X, Y, random_state=randomState, test_size=testSize, stratify=Y)

scoringConfig = {"precision": metrics.make_scorer(metrics.precision_score, average="binary", zero_division=0),
                 "recall": metrics.make_scorer(metrics.recall_score, average="binary", zero_division=0),
                 "f1": metrics.make_scorer(metrics.f1_score, average="binary", zero_division=0),
                 "accuracy": metrics.make_scorer(metrics.accuracy_score),
                 }

models = [
    # (linear_model.LogisticRegression(random_state=randomState),
    #  {
    #     'solver': ['saga'],
    #     'penalty': ['l2', 'l1'],
    #     'class_weight': [None, {0: .9, 1: .1}, 'balanced']
    # }),
    (ensemble.AdaBoostClassifier(random_state=randomState), {
        "n_estimators": [50]
    }),
    (ensemble.GradientBoostingClassifier(random_state=randomState), {
        "n_estimators": [100]
    }),
    # ((ensemble.RandomForestClassifier(random_state=randomState)),
    #  {
    #     'class_weight': [None, {0: .9, 1: .1}, 'balanced', 'balanced_subsample']
    # }),
    # (neural_network.MLPClassifier(random_state=randomState), {
    #     'activation': ['relu', 'tanh', 'logistic']
    # })
]

for model, paramGrid in models:
    print(f"Running - {model}")
    clf = model_selection.GridSearchCV(
        estimator=model, param_grid=paramGrid, scoring=scoringConfig,
        n_jobs=10, cv=model_selection.StratifiedKFold(n_splits=10), refit=target)

    clf.fit(trainX, trainY)

    results = pd.DataFrame(clf.cv_results_)
    results.sort_values(by=f"mean_test_{target}").to_csv(
        f"results/trial_{utils.getClassName(model)}_results.csv")

    predictedY = clf.predict(testX)

    cm = metrics.confusion_matrix(testY, predictedY)

    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    print(cm)

    acc = metrics.accuracy_score(testY, predictedY)
    prec = metrics.precision_score(testY, predictedY)
    rec = metrics.recall_score(testY, predictedY)
    f1 = metrics.f1_score(testY, predictedY)
    roc = metrics.roc_auc_score(testY, predictedY)

    print(f"acc: {acc} - prec: {prec} - rec: {rec} - f1: {f1} - rocauc: {roc}")

    display = metrics.RocCurveDisplay.from_estimator(
        clf, testX, testY, name=utils.getClassName(model))
    display.plot()

plt.plot()
