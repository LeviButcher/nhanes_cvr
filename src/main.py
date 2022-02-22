import pandas as pd
import Utils

# TODO
# Need to convert mortstat data to correct data types INT

NHANSE_YEARS = [(2001, 2002), (2003, 2004), (2005, 2006)]


def get_mortstat_data(years):
    data = map(Utils.compose(Utils.read_mortstat,
               Utils.build_morstat_download_url), years)
    return pd.concat(data)


def get_NHANSE_data(years):
    return 5


data = get_mortstat_data(NHANSE_YEARS)

print(data.head)
print(data.seqn.describe())
print(data.ucod_leading.count())
