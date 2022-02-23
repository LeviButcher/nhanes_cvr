import pandas as pd
import functools
import numpy as np

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


def toUpperCase(l):
    return list(map(lambda l: l.upper(), l))


mortality_colnames = toUpperCase(["publicid",
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
                                  "sa_wgt_new"])

drop_columns = toUpperCase(["publicid",
                            "dodqtr",
                            "dodyear",
                            "wgt_new",
                            "sa_wgt_new"])


def read_mortstat(location):
    # location = PATH | URL
    data = pd.read_fwf(location, widths=mortality_widths)
    data.columns = mortality_colnames
    data = data.assign(SEQN=data.PUBLICID.apply(lambda x: str(x)[:8]))
    data = functools.reduce(lambda acc, x: acc.drop(
        x, axis=1), drop_columns, data)

    # Convert data to correct types
    # https://www.cdc.gov/nchs/data/datalinkage/public-use-2015-linked-mortality-files-data-dictionary.pdf
    data.SEQN = pd.to_numeric(data.SEQN, errors="coerce", downcast="float")
    data.MORTSTAT = pd.to_numeric(
        data.MORTSTAT, errors="coerce")
    data.DIABETES = pd.to_numeric(
        data.DIABETES, errors="coerce")
    data.HYPERTEN = pd.to_numeric(
        data.HYPERTEN, errors="coerce")

    return data.set_index("SEQN")


def build_morstat_download_url(year):
    s, e = year
    return f"https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_{s}_{e}_MORT_2015_PUBLIC.dat"


def compose(f, g):
    return lambda x: f(g(x))


def get_mortstat_data(years):
    data = map(compose(read_mortstat,
               build_morstat_download_url), years)
    return pd.concat(data)


def get_nhanse_data(year_data_files):
    data = []
    for year, data_files in year_data_files:
        data.append([read_nhanse_data(build_nhanse_url(year, f))
                    for f in data_files])

    data = [functools.reduce(lambda acc, x: acc.merge(
        x, on="SEQN", how="inner"), l) for l in data]

    return pd.concat(data)


def build_nhanse_url(year, data_file):
    s, e = year
    return f"https://wwwn.cdc.gov/Nchs/Nhanes/{s}-{e}/{data_file}"


def read_nhanse_data(location):
    # location = PATH | URL
    return pd.read_sas(location).set_index('SEQN')


def get_nhanse_mortality_dataset(year_data_files):
    nhanse_years = [y for y, _ in year_data_files]
    nhanse_data = get_nhanse_data(year_data_files)
    mortality_data = get_mortstat_data(nhanse_years)
    return nhanse_data.join(mortality_data, on="SEQN", how="inner")
