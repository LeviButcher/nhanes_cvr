from sklearn import model_selection, linear_model, datasets, ensemble
import pandas as pd
from imblearn import under_sampling, pipeline

X, Y = datasets.load_iris(return_X_y=True)

models = [
    linear_model.LogisticRegression(), ensemble.RandomForestClassifier()
]
samplers = [
    under_sampling.ClusterCentroids(), under_sampling.CondensedNearestNeighbour()
]

conf = {
    "model": models,
    "sampler": samplers
}


pipe1Sampler = pipeline.Pipeline([
    ("sampler", samplers[0]),
    ("model", models[0])
])

conf = {
    "model": models,
    "sampler1": samplers,
    "sampler2": samplers
}

pipe2Sampler = pipeline.Pipeline([
    ("sampler1", samplers[0]),
    ("sampler2", samplers[1]),
    ("model", models[0])
])

gsCV = model_selection.GridSearchCV(
    estimator=pipe2Sampler, param_grid=conf, cv=10)

gsCV.fit(X, Y)

res = pd.DataFrame(gsCV.cv_results_)

print(res.params)
