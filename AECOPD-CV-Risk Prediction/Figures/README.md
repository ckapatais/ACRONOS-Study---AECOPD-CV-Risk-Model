# Figures

This folder contains all figures generated during the development, internal validation, external validation, pilot implementation evaluation, and sensitivity analyses of the AECOPD-CV prediction model.

Figures were produced directly from the corresponding analysis scripts and were not manually modified after generation. The outputs include discrimination (ROC) analyses, calibration plots, decision curve analyses (DCA), risk-stratification visualizations, coefficient visualizations, component-level sensitivity analyses, implementation-evaluation figures, and supplementary methodological figures.

The figures were generated using the following analysis workflows:

---

## Internal Validation

Development-cohort analyses used for model development and internal validation, including:

- Model performance assessment
- ROC analyses
- Calibration analyses
- Decision curve analyses
- Risk-stratification analyses
- Predictor coefficient visualizations
- Model comparison analyses

---

## MIMIC-IV External Validation

External validation of the frozen AECOPD-CV model using the MIMIC-IV database, including:

- 6-hour laboratory extraction analyses
- 12-hour laboratory extraction analyses
- 24-hour laboratory extraction analyses
- Anytime laboratory extraction analyses
- Laboratory-availability sensitivity analyses
- Calibration analyses
- Decision curve analyses
- Risk-stratification analyses

---

## eICU External Validation

External validation of the frozen AECOPD-CV model using the eICU Collaborative Research Database, including:

- 6-hour laboratory extraction analyses
- 12-hour laboratory extraction analyses
- 24-hour laboratory extraction analyses
- Anytime laboratory extraction analyses
- Calibration analyses
- Decision curve analyses
- Risk-stratification analyses

---

## Independent Temporal Pilot Implementation

Independent temporal pilot implementation evaluation of the frozen AECOPD-CV model, including:

- ROC analyses
- Predicted-risk distribution analyses
- Outcome-stratified risk visualizations
- LOWESS-based risk trend visualizations
- Supplementary implementation-performance figures

---

## Imputation Sensitivity Analyses

Evaluation of model robustness under alternative missing-data handling strategies, including:

- Median imputation
- Mean imputation
- K-nearest-neighbour imputation
- Iterative imputation
- Complete-case analysis

---

## Reproducibility

All figures were generated using fully reproducible scripted procedures implemented in Python. The corresponding scripts are available in the repository folders dedicated to:

- Internal Validation
- MIMIC-IV External Validation
- eICU External Validation
- Independent Temporal Pilot Implementation
- Imputation Sensitivity Analyses

No figures were manually modified after generation.
