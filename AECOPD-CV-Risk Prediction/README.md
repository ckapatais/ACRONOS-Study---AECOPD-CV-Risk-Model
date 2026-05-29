# AECOPD-CV Risk Model

This repository contains the code, validation workflows, figures, and web calculator associated with the development and validation of the AECOPD-CV prediction model for cardiovascular complications among patients hospitalized with acute exacerbation of chronic obstructive pulmonary disease (AECOPD).

The repository includes the complete analytical framework used for model development, internal validation, external validation, and sensitivity analyses.

## Repository Structure

### Internal Validation

Contains the scripts used for model development and internal validation within the derivation cohort, including model coefficient estimation, calibration assessment, discrimination analysis, decision curve analysis, and risk stratification.

### MIMIC-IV External Validation

Contains the scripts used for external validation of the frozen AECOPD-CV model within the MIMIC-IV database.

The following laboratory extraction strategies are included:

* 6-hour laboratory window (primary external validation analysis)
* 12-hour laboratory window
* 24-hour laboratory window
* Anytime laboratory extraction during hospitalization

Across all analyses, frozen model coefficients, frozen development-cohort median-imputation values, and frozen internally derived calibration parameters were applied without model refitting or external recalibration.

### eICU External Validation

Contains the scripts used for external validation of the frozen AECOPD-CV model within the eICU Collaborative Research Database.

The following laboratory extraction strategies are included:

* 6-hour laboratory window
* 12-hour laboratory window
* 24-hour laboratory window
* Anytime laboratory extraction during ICU admission

These analyses were performed to evaluate the robustness of model performance under varying conditions of laboratory availability within a heterogeneous multicentre ICU population.

### Imputation Sensitivity Analysis

Contains the scripts used to evaluate model robustness under alternative missing-data handling strategies, including:

* Median imputation
* Mean imputation
* K-nearest-neighbour imputation
* Iterative imputation
* Complete-case analysis

Performance was compared across discrimination, calibration, clinical utility, and risk-stratification metrics.

### Figures

Contains all figures generated from the internal validation, external validation, and sensitivity-analysis workflows.

### Web Calculator

Contains the standalone implementation of the AECOPD-CV risk calculator.

---

## Final Model Predictors

The final AECOPD-CV model includes six predictors:

* Age
* History of heart failure
* History of atrial fibrillation
* Blood gas pH
* Urea
* Lactate

---

## Outcome Definition

The primary outcome was a composite cardiovascular event occurring during hospitalization, defined as the occurrence of at least one of the following:

* Myocardial infarction
* Pulmonary embolism
* Pulmonary edema / acute heart failure decompensation phenotype
* Acute arrhythmia

---

## Data Sources

### Development and Internal Validation Cohort

Prospectively collected patients admitted with acute exacerbation of chronic obstructive pulmonary disease.

### External Validation Cohorts

* MIMIC-IV database
* eICU Collaborative Research Database

Access to both databases requires credentialed access through PhysioNet and completion of the required training and data-use agreements.

https://physionet.org/

---

## Reproducibility

The repository provides:

* Model-development scripts
* Internal-validation scripts
* MIMIC-IV external-validation scripts
* eICU external-validation scripts
* Imputation sensitivity-analysis scripts
* Figure-generation workflows
* Patient-level prediction outputs
* Model coefficients
* Standalone web calculator

All analyses were implemented using reproducible scripted workflows.

---

## Disclaimer

This repository and associated calculator are intended for research and educational purposes only. The AECOPD-CV model is designed to support clinical decision-making and should not replace clinical judgment, institutional protocols, or specialist evaluation.

---

## Citation

If you use this repository, please cite the associated manuscript and repository DOI.
