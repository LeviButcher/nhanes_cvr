import utils
import nhanse_dl
import numpy as np
import pandas as pd


def test_combine_data_frames():
    # Generate 3 data frames - a, b, c
    # combine all 3 data frames, call it o
    # o length should be equal to max(a, b, c)
    # o cols should be equal to the unique . map cols $ [a,b,c]
    # Should still work if as long as a,b,c has same index

    a = pd.DataFrame(
        {"seqn": np.arange(10), "A": np.arange(10)}).set_index("seqn")
    b = pd.DataFrame(
        {"seqn": np.arange(12), "B": np.arange(12)}).set_index("seqn")
    c = pd.DataFrame(
        {"seqn": np.arange(15), "C": np.arange(15), "D": np.arange(15)}).set_index("seqn")
    l = [("a", a), ("b", b), ("c", c)]

    print(dir(a))

    o = nhanse_dl.combine_data_frames(l)
    uniqueCols = np.unique([c for _, x in l for c in x.columns])

    assert o.shape[0] == max([x.shape[0] for _, x in l])
    assert all(o.columns == uniqueCols)

# pytest -k test_combine_df_columns


def test_combine_df_columns():
    # o should have only the columns passed in

    a = pd.DataFrame(
        {"seqn": np.arange(10), "A": np.arange(10), "B": np.arange(10), "C": np.arange(10)}).set_index("seqn")

    combine_directions = [(["A", "B", "C"], "D"), (["A", "A"], "E")]
    o_cols = [x for _, x in combine_directions]

    o = utils.combine_df_columns(combine_directions, a)

    assert all(o.columns == o_cols)
    assert o.shape == (a.shape[0], len(o_cols))
    assert all(o.D == a.A)
    assert all(o.E == a.A)


def test_combine_df_columns_with_NAN():
    a = pd.Series([1, 2, np.NAN, 4, np.NAN])
    b = pd.Series([1, np.NAN, 2, 3, np.NAN])
    c = pd.DataFrame({"a": a, "b": b})

    comb_dir = [(["a", "b"], "c"), (["a"], "a")]

    o = utils.combine_df_columns(comb_dir, c)

    expected = pd.DataFrame({"c": [1, 2, 2, 4, np.NAN], "a": a})

    assert expected.equals(o)
