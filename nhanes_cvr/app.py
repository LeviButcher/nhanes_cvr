import nhanes_cvr.utils as utils
import seaborn as sns
import nhanes_cvr.hypertenPaperRun as hypertenPaper
import nhanes_cvr.cvrAllRun as cvrAll

# Matplotlib/Seaborn Theming
sns.set_theme(style='darkgrid', palette='pastel')

saveDir = "results"

dataset = utils.get_nhanes_dataset()

dataset.dtypes.to_csv(f"{saveDir}/feature_types.csv")
# remove any unparsed features
dataset = dataset.select_dtypes(exclude=['object'])
dataset = dataset.loc[:500, :]
print(dataset.shape)


# Risk Analyses Runs

# hypertenPaper.runHypertensionRiskAnalyses(dataset, f"{saveDir}/hypertenPaper")
cvrAll.runCVRAllRiskAnalyses(dataset, f"{saveDir}/cvrAll")

# TODO Message Dr.A clarifying method
# TODO Fix testResult plot and ConfusionMatrix plot
# TODO Something is going on with CorrelationSelection
# Doing last if time
# TODO set up outputting choosen features for a bestModel within a labeller
