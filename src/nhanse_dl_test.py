import pandas as pd
import numpy as np
import nhanse_dl
import utils


def test_check_null_addition():
    x = pd.DataFrame({"a": [pd.NA, 2]})
    y = pd.DataFrame({"a": [2, pd.NA]})
    z = x + y

    print(z)

    assert any(z.isna())


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


def test_get_nhanse_data():
    # getting nhanse data for two years should return non-zero results

    nhanse_requests = [
        nhanse_dl.NHANSERequest(
            (2005, 2006), ["TCHOL_D.XPT", "TRIGLY_D.XPT", "HDL_D.xpt", "GLU_D.xpt"]),
        nhanse_dl.NHANSERequest(
            (2007, 2008), ["TCHOL_E.XPT",
                           "TRIGLY_E.XPT", "HDL_E.xpt", "GLU_E.xpt"]
        )
    ]

    res = nhanse_dl.get_nhanse_data(nhanse_requests)

    assert len(res) != 0


def test_get_mortality_data():
    nhanse_requests = [
        nhanse_dl.NHANSERequest(
            (2005, 2006), ["TCHOL_D.XPT", "TRIGLY_D.XPT", "HDL_D.xpt", "GLU_D.xpt"]),
        nhanse_dl.NHANSERequest(
            (2007, 2008), ["TCHOL_E.XPT",
                           "TRIGLY_E.XPT", "HDL_E.xpt", "GLU_E.xpt"]
        )
    ]

    res = nhanse_dl.get_mortstat_data(nhanse_requests)

    assert len(res) != 0

# pytest -k test_combine_df_columns


def test_combine_df_columns():
    # o should have only the columns passed in

    a = pd.DataFrame(
        {"seqn": np.arange(10), "A": np.arange(10), "B": np.arange(10), "C": np.arange(10)}).set_index("seqn")

    combine_directions = [(["A", "B", "C"], "D"), (["A", "A"], "E")]
    o_cols = [x[1] for x in combine_directions]

    o = utils.combine_df_columns(combine_directions, a)

    assert all(o.columns == o_cols)
    assert o.D.sum() == (a.A + a.B + a.C).sum()
    assert o.E.sum() == (a.A + a.A).sum()
