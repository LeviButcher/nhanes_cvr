from typing import List, Tuple
import pandas as pd
import utils


class NHANSERequest:
    year: Tuple
    files: List[str]

    def __init__(self, year, files):
        self.year = year
        self.files = files

    def __str__(self):
        return f"{self.year}:{self.files}"


mortality_colspecs = [(1, 14),
                      (15, 15),
                      (16, 16),
                      (17, 19),
                      (20, 20),
                      (21, 21),
                      (22, 22),
                      (23, 26),
                      (27, 34),
                      (35, 42),
                      (43, 45),
                      (46, 48)]

mortality_widths = [e - (s-1) for s, e in mortality_colspecs]

mortality_colnames = utils.toUpperCase(["publicid",
                                        "eligstat",
                                        "mortstat",
                                        "ucod_leading",
                                        "diabetes",
                                        "hyperten",
                                        "dodqtr",
                                        "dodyear",
                                        "wgt_new",
                                        "sa_wgt_new",
                                        "permth_int",
                                        "permth_exm"
                                        ])

drop_columns = utils.toUpperCase(["publicid",
                                  "dodqtr",
                                 "dodyear",
                                  "wgt_new",
                                  "sa_wgt_new"])


def read_mortstat(location: str) -> pd.DataFrame:
    # location = PATH | URL
    data = pd.read_fwf(location, widths=mortality_widths)
    data.columns = mortality_colnames
    return data.assign(SEQN=data.PUBLICID).drop(columns=drop_columns).apply(
        lambda x: pd.to_numeric(x, errors="coerce")).set_index("SEQN")


def build_morstat_download_url(year: Tuple[int, int]) -> str:
    s, e = year
    return f"https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/NHANES_{s}_{e}_MORT_2015_PUBLIC.dat"


def get_mortstat_data(nhanse_requests: List[NHANSERequest]) -> pd.DataFrame:
    years = [x.year for x in nhanse_requests]
    data = map(utils.compose(read_mortstat,
                             build_morstat_download_url), years)
    return pd.concat(data)


def get_nhanse_data(nhanse_requests: List[NHANSERequest]) -> pd.DataFrame:
    data = [[read_nhanse_data(build_nhanse_url(x.year, f)) for f in x.files]
            for x in nhanse_requests]

    data = [pd.concat(l, join="outer", axis=1) for l in data]
    # Older Nhanse dataset repeat columns
    data = [x.loc[:, ~x.columns.duplicated()] for x in data]
    return pd.concat(data)


def build_nhanse_url(year: Tuple[int, int], data_file: str) -> str:
    s, e = year
    return f"https://wwwn.cdc.gov/Nchs/Nhanes/{s}-{e}/{data_file}"


def read_nhanse_data(location: str) -> pd.DataFrame:
    # location = PATH | URL
    return pd.read_sas(location).set_index('SEQN')


def get_nhanse_mortality_dataset(nhanse_requests: List[NHANSERequest]) -> pd.DataFrame:
    nhanse_data = get_nhanse_data(nhanse_requests)
    mortality_data = get_mortstat_data(nhanse_requests)
    return nhanse_data.join(mortality_data, on="SEQN", how="outer")
