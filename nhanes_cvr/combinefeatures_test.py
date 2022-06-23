import numpy as np
import nhanes_cvr.combinefeatures as cf
import pandas as pd


def test_meanCombine_simple():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    X = pd.DataFrame([s, s, s, s, s])
    expected = pd.Series([s.mean(), s.mean(), s.mean(), s.mean(), s.mean()])
    res = cf.meanCombine(X)

    assert res.equals(expected)


def test_meanCombine_complex():
    expected = pd.Series([13/4, 28/4, 35/4])
    X = pd.DataFrame([
        [5, 2, 1, 5],
        [10, 4, 9, 5],
        [15, 8, 7, 5]
    ])
    res = cf.meanCombine(X)

    assert res.equals(expected)


def test_firstNonNullCombine():
    X = pd.DataFrame([
        [2, 3, 4, 5],
        [np.NAN, None, np.NAN, 4],
        [None, np.NAN, 2, np.NAN]
    ])

    res = cf.firstNonNullCombine(X)
    expected = pd.Series([2.0, 4.0, 2.0])

    assert res.equals(expected)


class TestRunCombine:
    def test_runCombine(self):
        conf = cf.CombineFeatures(
            ["A"], "C", lambda x: x.A, cf.noPostProcessing)

        X = pd.DataFrame({"A": [1, 2, 3, 4, 5]})

        res = cf.runCombine(conf, X)

        assert res.shape == (5,)
        assert res.equals(X.A)
        assert res.name == "C"

    def test_runCombine_complex(self):
        conf = cf.CombineFeatures(
            ["A", "B", "A"], "D", cf.meanCombine, cf.noPostProcessing)

        X = pd.DataFrame({"A": [1, 2, 3, 4, 5],
                          "B": [5, 4, 3, 2, 1],
                          "C": [5, 5, 5, 5, 5]})

        expected = X.loc[:, conf.features].mean(axis=1)

        res = cf.runCombine(conf, X)

        assert res.shape == (5,)
        assert res.equals(expected)
        assert res.name == "D"

    def test_runCombine_postProcess(self):
        conf = cf.rename("A", "Z", postProcess=cf.meanMissingReplacement)
        X = pd.DataFrame({"A": [5, pd.NA, 5, pd.NA, 5]})
        res = cf.runCombine(conf, X)

        assert res.equals(pd.Series([5.0, 5.0, 5.0, 5.0, 5.0]))

    def test_runCombines(self):
        configs = [
            cf.rename("A", "Z", cf.noPostProcessing),
            cf.CombineFeatures(["A", "B"], "A+B",
                               lambda X: X.A + X.B, cf.noPostProcessing)
        ]
        X = pd.DataFrame({"A": [1, 2, 3, 4, 5],
                          "B": [5, 4, 3, 2, 1],
                          "C": [5, 5, 5, 5, 5]})

        res = cf.runCombines(configs, X)
        expected = pd.DataFrame({"Z": X.A, "A+B": X.A + X.B})

        assert res.equals(expected)
