import numpy as np
import pandas as pd
from sklearn import linear_model, model_selection, utils

# New Plan: Need to make two custom classes
# BalancedKFold - balance dataset into equal class 50/50 then performs k-fold
# RepeatedBalancedKFold - Performs balancedkfold n many times


def balanceDataset(X, y, random_state):
    # Equalize the amount of class instance by lows class in y
    # TODO: Not hardcode what class labels we look for
    # should flow like so:
    #   findMinClassCount -> bucket datasets into classes -> sample each set by minCount -> combine and return

    trueCount = (y == 1).sum()
    trueSet = X.loc[y == 1, :].index
    falseSet = X.loc[y == 0, :].sample(
        n=trueCount, random_state=random_state).index

    indexes = np.concatenate((trueSet._values, falseSet._values))

    return X.iloc[indexes, :], y.iloc[indexes]


class BalancedKFold(model_selection.KFold):

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y=None, groups=None):
        X, y = balanceDataset(X, y, self.random_state)

        return super().split(X, y, groups)


class RepeatedBalancedKFold(model_selection.RepeatedKFold):
    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        super(model_selection.RepeatedKFold, self).__init__(
            BalancedKFold, n_repeats=n_repeats, random_state=random_state, n_splits=n_splits)


m = linear_model.LinearRegression()
X = pd.DataFrame(np.random.rand(10, 10))
# y = np.random.randint(2, size=10)
y = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])


cv = BalancedKFold()

res = model_selection.cross_validate(m, X, y, cv=cv, return_train_score=True)

print(res)
