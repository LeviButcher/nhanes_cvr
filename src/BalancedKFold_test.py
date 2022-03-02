from posixpath import split
from BalancedKFold import BalancedKFold, RepeatedBalancedKFold
import numpy as np
import pandas as pd


def test_BalancedKFold_splitShouldEqualDoubleSmallestClass():
    X = pd.DataFrame(np.random.rand(100, 100))
    Y = pd.Series(np.repeat([0, 0, 0, 1, 1], 20))
    smallestCount = (Y == 1).sum()
    splits = 10

    cv = BalancedKFold(n_splits=splits, shuffle=True, random_state=42)

    for train, test in cv.split(X, Y):
        assert smallestCount * 2 == len(train) + len(test)


def test_BalancedKFold_noFoldEqualsPreviousFold():
    X = pd.DataFrame(np.random.rand(100, 100))
    Y = pd.Series(np.repeat([0, 0, 0, 1, 1], 20))
    splits = 10

    cv = BalancedKFold(n_splits=splits, shuffle=True, random_state=42)

    prevTrain, prevTest = None, None

    for train, test in cv.split(X, Y):
        if prevTrain is not None:
            assert (prevTrain != train).any()
            assert (prevTest != test).any()

        prevTrain = train
        prevTest = test


# Test could be better but it'll do
def test_RepeatedBalancedKFold_randomizedSetsEachKFold():
    # Each 10 folds should have different data then every other 10 folds (Randomness could make this fail)
    X = pd.DataFrame(np.random.rand(100, 100))
    Y = pd.Series(np.repeat([0, 0, 0, 1, 1], 20))
    splits = 10
    repeats = 10

    cv = RepeatedBalancedKFold(
        n_splits=splits, n_repeats=repeats, random_state=42)

    folds = [np.concatenate((train, test)) for train, test in cv.split(X, Y)]

    prev10Split = None
    equaled = []
    for splitIndexes in folds[::10]:
        if prev10Split is not None:
            equaled.append((prev10Split == splitIndexes).all())

        prev10Split = splitIndexes

    # Can be the same once or twice but they can't all be the same
    # assert equaled is None
    assert not all(equaled)
