import nhanes_cvr.utils as utils
import seaborn as sns
import nhanes_cvr.hypertenPaperRun as hypertenPaper
import nhanes_cvr.cvrAllRun as cvrAll
import nhanes_cvr.cvrHandpickedRun as cvrHandpicked

# Matplotlib/Seaborn Theming
sns.set_theme(style='darkgrid', palette='pastel')

saveDir = "results"

dataset = utils.get_nhanes_dataset()

dataset.dtypes.to_csv(f"{saveDir}/feature_types.csv")
# remove any unparsed features
dataset = dataset.select_dtypes(exclude=['object'])

# Risk Analyses Runs

# cvrAll.runCVRAllRiskAnalyses(dataset, f"{saveDir}/cvrAll")
hypertenPaper.runHypertensionRiskAnalyses(dataset, f"{saveDir}/hypertenPaper")

# Two Issues
# Can't Do handpicked because inbedded within dataframe - needs pandas column
# Can't do picked features of selection because need dataframe
