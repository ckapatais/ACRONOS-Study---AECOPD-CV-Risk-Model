Section 1. Overview:

This repository contains the final internal-validation workflow for the AECOPD-CV prediction model developed to estimate the risk of acute cardiovascular complications among ICU patients admitted with acute exacerbation of chronic obstructive pulmonary disease (AECOPD).
The script implements the complete internal-validation pipeline used for the analyses reported in the manuscript and supplementary appendix, including:

1. Predictor extraction.
2. Median-imputation handling.
3. Logistic-regression model fitting.
4. Calibration assessment.
5. Discrimination analysis.
6. Decision-curve analysis.
7. Quartile-based risk stratification.
8. Figure/table generation.

The workflow was implemented using a predefined and reproducible scripted pipeline.

Section 2. Validation Framework:
Internal validation was performed using a stratified train–test split approach. The modelling workflow consisted of:

1. Stratified division of the dataset into training and held-out test subsets.
2. Median imputation fitted exclusively within the training subset.
3. Logistic-regression model fitting within the training subset.
4. Prediction generation within the held-out test subset.
5. Estimation of calibration intercept and calibration slope parameters within the internal-validation framework.
6. Evaluation of discrimination, calibration, clinical utility, and risk stratification performance.

Bootstrap resampling procedures were used to estimate confidence intervals for discrimination metrics.

Section 3. Predictors Included in the Final Model:
The final internally validated model included the following predictors:
1. Age
2. History of heart failure
3. History of atrial fibrillation
4. Arterial pH
5. Urea
6. Lactate

Continuous predictors were retained in their native scale and transformed according to the predefined model specification during odds-ratio estimation and figure generation.

Section 4. Outcome Definition:
The composite cardiovascular outcome included the occurrence of at least one of the following events during ICU admission:
1. Myocardial infarction.
2. pulmonary embolism.
3. pulmonary edema.
4. acute arrhythmia.

The composite outcome variable was generated programmatically within the script using predefined event mappings.

Section 5. Missing-Data Handling:
Missing predictor values were handled using median imputation. Median values were estimated exclusively within the training subset and subsequently applied unchanged to the held-out test subset during internal validation.
No information from the held-out test subset was used during model fitting or imputation estimation.

Section 6. Generated Outputs:
The script automatically generates:
- Figures
- ROC curve
- Calibration plot
- Decision curve analysis
- Quartile event-rate plot
- Forest plot of model coefficients
- Composite multi-panel figure
- Tables / CSV outputs
- Model coefficients
- ROC metrics
- Calibration metrics
- Quartile summaries
- Decision-curve values
- Variable availability summaries
- Patient-level predictions
- Internal-validation performance summaries
- Additional outputs
- Frozen calibration intercept/slope parameters
- Variable mapping files
- Event mapping files

Section 7. Main Output Files
Key generated files include:
| File                                 | Description                                      |
| ------------------------------------ | ------------------------------------------------ |
| summary_internal_validation.csv      | Overall internal-validation performance summary  |
| model_coefficients_for_figures.csv   | Final regression coefficients                    |
| table_coefficients_exact.csv         | Odds ratios and confidence intervals             |
| patient_level_figure_inputs.csv      | Patient-level predictions                        |
| recalibration_parameters.json        | Frozen internally derived calibration parameters |
| Internal_Validation_Composite.png    | Composite multi-panel validation figure          |

Section 8. Software Environment
The script was developed and tested using:

1. Python 3.11.9
2. Pandas
3. Numpy
4. Statsmodels
5. Scikit-learn
6. Matplotlib
7. Scipy
8. Pillow

Section 9. Reproducibility:
All analyses were implemented using scripted procedures to ensure reproducibility and transparency.
The repository contains the exact code used for:
1. Internal validation.
2. Figure generation.
3. Export of patient-level predictions and performance metrics.

No manual post-processing steps were applied after script execution.
