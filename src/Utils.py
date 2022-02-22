import pandas as pd
import functools

mortality_colspecs = [(0, 13),
                      (14, 14),
                      (15, 15),
                      (16, 18),
                      (19, 19),
                      (20, 20),
                      (21, 21),
                      (22, 25),
                      (26, 33),
                      (34, 41),
                      (42, 44),
                      (45, 47)]

mortality_widths = [14, 1, 1, 3, 1, 1, 1, 4, 3, 3, 5, 6]

mortality_colnames = ["publicid",
                      "eligstat",
                      "mortstat",
                      "ucod_leading",
                      "diabetes",
                      "hyperten",
                      "dodqtr",
                      "dodyear",
                      "permth_int",
                      "permth_exm",
                      "wgt_new",
                      "sa_wgt_new"]

drop_columns = ["publicid",
                "dodqtr",
                "dodyear",
                "wgt_new",
                "sa_wgt_new"]


def read_mortstat(path):
    data = pd.read_fwf(path, widths=mortality_widths)
    data.columns = mortality_colnames
    data = data.assign(seqn=data.publicid.apply(lambda x: str(x)[0:4]))
    data = functools.reduce(lambda acc, x: acc.drop(
        x, axis=1), drop_columns, data)
    return data


def build_morstat_download_url(year):
    s, e = year
    return f"https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_{s}_{e}_MORT_2015_PUBLIC.dat"


def compose(f, g):
    return lambda x: f(g(x))
