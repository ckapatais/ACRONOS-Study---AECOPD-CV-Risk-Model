# AECOPD-CV Web-Based Risk Calculator

This folder contains the standalone web-based implementation of the AECOPD-CV prediction model for cardiovascular complications among patients hospitalized with acute exacerbation of chronic obstructive pulmonary disease (AECOPD).

The calculator provides an interactive browser-based interface that allows users to enter patient characteristics and laboratory values and obtain an individualized predicted probability of experiencing an in-hospital cardiovascular complication.

The web calculator implements the final frozen AECOPD-CV model exactly as specified during model development and validation. No model retraining, recalibration, coefficient updating, or outcome-informed parameter estimation is performed within the application.

---

# Purpose

The calculator was developed to facilitate practical implementation of the AECOPD-CV model in clinical and research settings.

The platform provides:

- Individual patient risk prediction
- Automated predictor processing
- Risk-category assignment
- Confidence interval reporting
- Bedside score interpretation
- Educational information regarding model development and validation
- Access to methodological references and supporting documentation

The application operates entirely within the user's web browser and does not require installation of additional software, databases, or server-side components.

---

# Online Access

The AECOPD-CV calculator is publicly available online at:

https://aecopd-cv.com

The online version provides the same prediction engine implemented within the standalone HTML application contained in this repository.

Users may either:

- Access the calculator directly through the public website
- Run the standalone HTML version locally
- Deploy the standalone HTML version within institutional or research environments

The public deployment is intended to facilitate rapid access for clinicians, researchers, and healthcare professionals without requiring local installation.

---

# Model Predictors

The AECOPD-CV model uses six predefined predictors:

- Age
- History of heart failure
- History of atrial fibrillation
- Arterial pH
- Urea
- Lactate

These predictors are entered directly through the calculator interface.

---

# Outcome Definition

The primary outcome predicted by the calculator is a composite cardiovascular complication occurring during hospitalization, defined as the occurrence of at least one of the following:

- Myocardial infarction
- Pulmonary embolism
- Pulmonary edema / acute heart failure decompensation phenotype
- Acute arrhythmia

---

# Calculator Components

The web platform contains five integrated sections.

## 1. Calculator

The main calculator interface allows users to:

- Enter patient age
- Enter arterial pH values
- Enter urea values
- Enter lactate values
- Specify the presence or absence of:
  - Heart failure
  - Atrial fibrillation

The calculator automatically:

- Applies the frozen AECOPD-CV model coefficients
- Generates predicted probabilities
- Calculates confidence intervals
- Assigns risk categories
- Displays graphical risk-position indicators

---

## 2. Bedside Score Guide

The score guide provides:

- Risk-category interpretation
- Bedside implementation guidance
- Risk-threshold information
- Clinical interpretation support

---

## 3. About the Model

The About section summarizes:

- Model rationale
- Predictor selection
- Outcome definition
- Development methodology
- Validation framework
- Intended use

---

## 4. References

The References section contains the scientific sources and methodological references supporting model development and validation.

---

## 5. Contact

The Contact section provides project and correspondence information associated with the AECOPD-CV model.

---

# File Structure

This implementation consists of a single standalone HTML file:

```text
aecopd_cv_risk_calculator.html
```

The file contains:

- HTML interface components
- Embedded CSS styling
- Embedded JavaScript prediction engine
- Risk visualization elements
- Educational content pages
- Reference and contact pages

No external databases are required.

No server infrastructure is required.

No third-party web services are required for prediction generation.

---

# How to Access

## Option 1: Online Calculator

Access the public calculator directly:

https://aecopd-cv.com

No installation is required.

---

## Option 2: Local Use

Download:

```text
aecopd_cv_risk_calculator.html
```

Open the file directly in:

- Google Chrome
- Microsoft Edge
- Mozilla Firefox
- Safari

---

## Option 3: Institutional Deployment

The calculator can be hosted directly through:

- GitHub Pages
- Institutional websites
- Research project websites
- Hospital intranets

Because the calculator is fully self-contained, deployment only requires uploading the HTML file.

---

# Validation Framework

The calculator implements the final frozen AECOPD-CV model that was evaluated through:

## Internal Validation

- Discrimination assessment
- Calibration assessment
- Decision curve analysis
- Risk-stratification analyses

## MIMIC-IV External Validation

- 6-hour laboratory window
- 12-hour laboratory window
- 24-hour laboratory window
- Anytime laboratory extraction window

## eICU External Validation

- 6-hour laboratory window
- 12-hour laboratory window
- 24-hour laboratory window
- Anytime laboratory extraction window

## Independent Temporal Pilot Implementation

Pilot implementation evaluation using an independent temporal cohort.

No model updating was performed during any validation stage.

---

# Reproducibility

The calculator represents a direct implementation of the final AECOPD-CV model and is provided alongside:

- Internal-validation scripts
- MIMIC-IV external-validation scripts
- eICU external-validation scripts
- Independent temporal pilot implementation scripts
- Imputation sensitivity-analysis scripts
- Figure-generation workflows
- Public web calculator (https://aecopd-cv.com)
- Standalone HTML calculator

The implementation is fully reproducible and corresponds to the final validated model version reported in the associated study.

---

# Disclaimer

This calculator is intended for research and educational purposes only.

The AECOPD-CV model is designed to support clinical decision-making and should not replace clinical judgment, institutional protocols, specialist consultation, or individualized patient assessment.

Predictions generated by the calculator should be interpreted within the broader clinical context of each patient.
