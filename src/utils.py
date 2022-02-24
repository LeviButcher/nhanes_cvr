from typing import List, Tuple
import pandas as pd
import re


def toUpperCase(l):
    return list(map(lambda l: l.upper(), l))


def compose(f, g):
    return lambda x: f(g(x))


def labelCauseOfDeathAsCVR(ucod_leading):
    # Different meanings of ucod_leading - https://www.cdc.gov/nchs/data/datalinkage/public-use-2015-linked-mortality-files-data-dictionary.pdf
    # 1 is Diesease of heart
    # 2 is Cerebrovascular Diseases
    return 1 if ucod_leading == 1 or ucod_leading == 5 else 0


def avg(l):
    return sum(l) / len(l)


CombineDirections = Tuple[List[str], str]


def combine_df_columns(combine_directions: List[CombineDirections], x: pd.DataFrame) -> pd.DataFrame:
    def combine(x: pd.DataFrame, d: CombineDirections):
        cols, target = d
        series = x.loc[:, cols].sum(axis=1)
        return pd.DataFrame(series, columns=[target])

    res = [combine(x, d) for d in combine_directions]

    return pd.concat(res, axis=1)
