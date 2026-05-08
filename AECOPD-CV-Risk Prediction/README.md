# AECOPD-CV Risk Model

This repository contains the code and web calculator associated with the ACRONOS study, which developed and externally validated a clinical prediction model for cardiovascular complications during hospitalization for acute exacerbation of chronic obstructive pulmonary disease (AECOPD).

## Repository contents

- `internal_validation.py` — model development and internal validation
- `mimic_external_validation.py` — external validation using the MIMIC-IV database
- `eicu_external_validation.py` — external validation using the eICU Collaborative Research Database
- `aecopd_cv_risk_calculator_singlefile_final.html` — standalone web calculator
- `requirements.txt` — Python package dependencies
- `LICENSE` — repository license information

## Model predictors

The final AECOPD-CV model includes:
- Age
- History of heart failure
- History of atrial fibrillation
- Arterial pH
- Urea
- Lactate

## Outcomes

The primary outcome was a composite cardiovascular event during index hospitalization, including:
- Myocardial infarction
- Pulmonary embolism
- Pulmonary edema
- Acute arrhythmia

## Data sources

### Derivation cohort
Prospectively collected hospitalized patients with AECOPD.

### External validation cohorts
- MIMIC-IV database
- eICU Collaborative Research Database

Access to MIMIC-IV and eICU requires credentialed access through PhysioNet and completion of the required data use agreements and training:
https://physionet.org/

## Reproducibility

The repository provides:
- full model coefficients
- validation scripts
- figure-generation workflows
- standalone calculator implementation

## Disclaimer

This repository and calculator are intended for research and educational purposes only. The model should support, not replace, clinical judgment. Clinical decisions should be made in conjunction with patient assessment, institutional protocols, and specialist evaluation where appropriate.

## Citation

If you use this repository, please cite the associated manuscript and repository DOI.
