from typing import Callable, List, Tuple, TypeVar, Union, Dict, NewType, Any
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn import pipeline
from sklearn import model_selection, preprocessing, linear_model, ensemble, impute, feature_selection

import pandas as pd

DF = pd.DataFrame
XSet = pd.DataFrame
YSet = pd.Series
XYPair = Tuple[XSet, YSet]

CVSearch = Union[model_selection.GridSearchCV,
                 model_selection.RandomizedSearchCV]
GridSearchDF = NewType('GridSearchDF', pd.DataFrame)
CVTrainDF = NewType('CVTrainDF', pd.DataFrame)
ScoreDF = NewType('ScoreDF', pd.DataFrame)
CVRes = Tuple[CVTrainDF, ScoreDF]
LabellerTrainDF = NewType('LabellerTrainDF', pd.DataFrame)
LabellerTestDF = NewType('LabellerTestDF', pd.DataFrame)
LabellerRes = Tuple[LabellerTrainDF, LabellerTestDF]


# Config Typing
Scoring = Dict[str, Any]
ModelConf = Dict[str, List[Any]]
Model = Union[linear_model.LogisticRegression,
              ensemble.RandomForestClassifier, pipeline.Pipeline]
Scaling = Union[preprocessing.StandardScaler, preprocessing.MinMaxScaler]

PipeLine = imblearn.pipeline.Pipeline
PipeLinConf = Dict
PipeLineCV = Tuple[str, PipeLine, PipeLinConf]


Fold = model_selection.StratifiedKFold
ModelConst = Callable[[], Model]
GenModelConf = Tuple[ModelConst, ModelConf]
GenModelConfList = List[GenModelConf]
Sampler = RandomUnderSampler
SamplerConst = Callable[[], Sampler]
SamplerList = List[Sampler]
SamplerConstList = List[SamplerConst]
# Function that Labels Dataset
Labeller = Callable[[pd.DataFrame], XYPair]
NamedLabeller = Tuple[str, Labeller]
NamedLabellerList = List[NamedLabeller]
# Function that selects features/samples to use for training
Selector = Callable[[XYPair], XYPair]
OutputSelector = Callable[[str], Selector]
NamedSelector = Tuple[str, Selector]
NamedOutputSelector = Tuple[str, OutputSelector]
NamedSelectors = List[NamedSelector]
NamedOutputSelectors = List[NamedOutputSelector]
NamedLabeller = Tuple[str, Labeller]
SelectorCVTestDF = pd.DataFrame
Replacement = impute.SimpleImputer
T = TypeVar('T')
Const = Callable[[], T]
ConstList = List[Const[T]]
Selection = Union[feature_selection.SelectFwe,
                  feature_selection.VarianceThreshold, feature_selection.SelectPercentile]

Folding = model_selection.StratifiedKFold
GetRisk = Callable[[pd.DataFrame], pd.Series]
