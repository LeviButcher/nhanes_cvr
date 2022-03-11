import functools
from typing import List, Tuple
import numpy as np
import pandas as pd


def toUpperCase(l):
    return list(map(lambda l: l.upper(), l))


def compose(f, g):
    return lambda x: f(g(x))


def labelCauseOfDeathAsCVR(ucod_leading):
    # Different meanings of ucod_leading - https://www.cdc.gov/nchs/data/datalinkage/public-use-2015-linked-mortality-files-data-dictionary.pdf
    # 1 is Diesease of heart
    # 2 is Cerebrovascular Diseases
    return 1 if ucod_leading == 1 or ucod_leading == 5 else 0


CombineDirections = Tuple[List[str], str]

# combine_df_columns
# combine_directions: List of directions of columns to combine into 1 columns
# x: Dataframe who columns to aggregate into the output of combine_directions
# returns a new dataframe whose only
#
# Output Properties:
#   - columns are name the second value of the tuple combine_directions
#   - Size of output should be unchanged
#   - If both columns that are combined are NAN then the results should be NAN


def combine_df_columns(combine_directions: List[CombineDirections], x: pd.DataFrame) -> pd.DataFrame:
    # Do a reduce here where we return the first real value
    def keep_first_non_NAN(x): return functools.reduce(
        lambda b, y: b if not np.isnan(b) else y, x, np.NAN)
    # pd.NA

    def combine(x: pd.DataFrame, d: CombineDirections):
        cols, target = d
        series = x.loc[:, cols].agg(keep_first_non_NAN, axis=1)
        return pd.DataFrame(series, columns=[target])

    res = [combine(x, d) for d in combine_directions]

    return pd.concat(res, axis=1)


def unique(list):
    return functools.reduce(
        lambda l, x: l if x in l else l + [x], list, [])
