import pandas as pd
import numpy as np
import nhanse_dl


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


def test_get_mortality_constructs_correctly():
    year = (1999, 2000)
    url = nhanse_dl.build_morstat_download_url(year)
    data = nhanse_dl.read_mortstat(url)

    expected = pd.DataFrame({"SEQN": [9960.0], "ELIGSTAT": [1.0],
                            "MORTSTAT": [1], "UCOD_LEADING": [2.0],
                             "DIABETES": [0.0], "HYPERTEN": [0.0],
                             "PERMTH_INT": [21.0], "PERMTH_EXM": [21.0]
                             }).set_index("SEQN")

    res = data.loc[9960, :]

    print(res)

    assert expected.loc[9960, :].equals(res)
