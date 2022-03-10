import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# from BalancedKFold import BalancedKFold

model = LogisticRegression()
cv = StratifiedKFold(
    n_splits=10, shuffle=True, random_state=42)

param_grid = {}

X = pd.DataFrame(np.random.randint(
    0, 100, size=(100, 4)), columns=list('ABCD'))
Y = pd.Series(np.random.choice([0, 1], 100))

gscv = GridSearchCV(model, param_grid, cv=cv, scoring=[
                    "precision", "recall", "f1"], refit="f1")

res = gscv.fit(X, Y)
print(res)
res = pd.DataFrame(res.cv_results_)
print(res.mean_test_precision)
