MIMIC-IV External Validation

This folder contains the scripts used for external validation of the AECOPD-CV prediction model in the MIMIC-IV database.

The external validation framework was implemented using a frozen-model approach. Original regression coefficients and predefined median-imputation values derived from the development cohort were applied directly to the MIMIC-IV dataset without model refitting, recalibration, or coefficient updating.

External validation analyses were performed using multiple laboratory extraction strategies to evaluate robustness under varying conditions of laboratory availability:
- 6-hour laboratory window
- 12-hour laboratory window
- 24-hour laboratory window
- Anytime laboratory extraction window

For each extraction strategy, laboratory predictors were defined using the earliest available laboratory value within the corresponding time window following ICU admission.

The validated model included the following predefined predictors:
- Age
- History of heart failure
- History of atrial fibrillation
- Arterial pH
- Urea
- Lactate

The composite cardiovascular outcome included:
- Myocardial infarction
- Pulmonary embolism
- Pulmonary edema
- Acute arrhythmia

All scripts were developed using fully reproducible computational workflows to ensure consistency across cohort construction, predictor extraction, prediction generation, and performance evaluation procedures.
