import numpy as np
import pandas as pd
from sklearn import linear_model, model_selection, utils

# TODO WRITE TESTS FOR THIS

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

    xSet = pd.concat([X.loc[falseSet, :], X.loc[trueSet, :]])
    ySet = pd.concat([y.loc[falseSet], y.loc[trueSet]])

    return xSet, ySet


class BalancedKFold(model_selection.KFold):

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y, groups=None):
        bX, by = balanceDataset(X, y, self.random_state)

        return super().split(bX, by, groups)


class RepeatedBalancedKFold(model_selection.RepeatedKFold):
    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        super(model_selection.RepeatedKFold, self).__init__(
            BalancedKFold, n_repeats=n_repeats, random_state=random_state, n_splits=n_splits)


# X = pd.DataFrame(np.random.rand(100, 100))
# Y = pd.Series(np.repeat([0, 0, 0, 1, 1], 20))


# cv = BalancedKFold(n_splits=10, shuffle=True, random_state=42)

# for train, test in cv.split(X, Y):
#     x = X.iloc[train, :]
#     y = Y.iloc[train]
#     print(x.shape)
