import nhanes_cvr.utils as utils
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

    o = utils.combine_data_frames(l)
    uniqueCols = np.unique([c for _, x in l for c in x.columns])

    assert o.shape[0] == max([x.shape[0] for _, x in l])
    assert all(o.columns == uniqueCols)
