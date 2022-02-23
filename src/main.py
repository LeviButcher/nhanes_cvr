import pandas as pd
import Utils
from sklearn import linear_model

# TODO
# [x] Need to convert mortstat data to correct data types INT
# [x] Figure out how to Combining NHANSE data
# [x] Check what the 6 lab measurements were and find there nhanse files
# [x] Enter all NHANSE Years with correct data files
# [] Validate if Dataset is correctly put together
# [x] Setup X,Y from dataset
# [] Setup 60,30,10 split for training,val, and testing
# [] Setup graphs of training results

# Forget about LDLC for now
markers = ["TotalChol", "LDL", "HDL", "FBG", "TG", "LDLC"]


NHANSE_DATA_FILES = [
    ((2005, 2006), ["TCHOL_D.XPT", "TRIGLY_D.XPT", "HDL_D.xpt", "GLU_D.xpt"]),
    ((2007, 2008), ["TCHOL_E.XPT", "TRIGLY_E.XPT", "HDL_E.xpt", "GLU_E.xpt"])]

# IS Fasting glucose different then glucose
# All measurements in (mg/dl)
features_descritions = [("LBXTC", "Total Chol"), ("LBDLDL", "LDL Chol"),
                        ("LBDHDD", "HDL Chol"), ("LBXGLU", "Fasting Glucose"), ("LBXTR", "Triglyceride")]
features = [x for x, _ in features_descritions]

dataset = Utils.get_nhanse_mortality_dataset(NHANSE_DATA_FILES)


def labelCauseOfDeathAsCVR(ucod_leading):
    # Different meanings of ucod_leading - https://www.cdc.gov/nchs/data/datalinkage/public-use-2015-linked-mortality-files-data-dictionary.pdf
    # 1 is Diesease of heart
    # 2 is Cerebrovascular Diseases

    match ucod_leading:
        case 0 | 1:
            return 1
        case _:
            return 0

    # print(dataset.columns)
X = dataset.loc[:, features].assign(
    CVR=dataset.UCOD_LEADING.apply(labelCauseOfDeathAsCVR)).dropna()
Y = X.CVR
X = X.loc[:, features]

# ML Steps
# Split dataset into train,val, and test
# Run Training on train set, show performance curve using val, KFold validation
# Run On test set showing F1 score, precision, and recall.

reg = linear_model.LinearRegression()
reg.fit(X, Y)

predictY = reg.predict(X)
print((predictY == Y).count())
