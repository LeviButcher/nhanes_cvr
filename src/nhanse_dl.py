from typing import List, Tuple
import pandas as pd
import functools
import utils


class NHANSERequest:
    year: Tuple
    files: List[str]

    def __init__(self, year, files):
        self.year = year
        self.files = files

    def __str__(self):
        return f"{self.year}:{self.files}"


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
mortality_colnames = utils.toUpperCase(["publicid",
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
drop_columns = utils.toUpperCase(["publicid",
                                  "dodqtr",
                                  "dodyear",
                                  "wgt_new",
                                  "sa_wgt_new"])


def read_mortstat(location: str) -> pd.DataFrame:
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


def build_morstat_download_url(year: Tuple[int, int]) -> str:
    s, e = year
    return f"https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_{s}_{e}_MORT_2015_PUBLIC.dat"


def get_mortstat_data(nhanse_requests: List[NHANSERequest]) -> pd.DataFrame:
    years = [x.year for x in nhanse_requests]
    data = map(utils.compose(read_mortstat,
               build_morstat_download_url), years)
    return pd.concat(data)


def get_nhanse_data(nhanse_requests: List[NHANSERequest]) -> pd.DataFrame:
    data = []
    for x in nhanse_requests:
        year, data_files = x.year, x.files
        data.append([(year, read_nhanse_data(build_nhanse_url(year, f)))
                    for f in data_files])

    data = [combine_data_frames(l) for l in data]

    return pd.concat(data)


def combine_data_frames(listOfFrames: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    def joinDataFrame(acc, x): return acc.join(
        x[1], how="outer", rsuffix=x[0])

    return functools.reduce(joinDataFrame, listOfFrames, pd.DataFrame())


def build_nhanse_url(year: Tuple[int, int], data_file: str) -> str:
    s, e = year
    return f"https://wwwn.cdc.gov/Nchs/Nhanes/{s}-{e}/{data_file}"


def read_nhanse_data(location: str) -> pd.DataFrame:
    # location = PATH | URL
    return pd.read_sas(location).set_index('SEQN')


def get_nhanse_mortality_dataset(nhanse_requests: List[NHANSERequest]) -> pd.DataFrame:

    nhanse_data = get_nhanse_data(nhanse_requests)
    mortality_data = get_mortstat_data(nhanse_requests)
    return nhanse_data.join(mortality_data, on="SEQN", how="inner")
