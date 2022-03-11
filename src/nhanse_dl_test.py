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
