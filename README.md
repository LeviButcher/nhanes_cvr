# NHANSE-CVR

Evaluating CardioRisk using NHANSE data

Morality Files: <https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality>

Scalers

- None - No Scaling
- preprocessing.MinMaxScaler() - Scaled via min/max to be between 0 - 1,
- preprocessing.Normalizer() - Scales to be in unit form,
- preprocessing.StandardScaler() - remove mean and scale to unit variance,
- preprocessing.RobustScaler() - removes the median and scales according to the quantile range (1st to 3rd)
