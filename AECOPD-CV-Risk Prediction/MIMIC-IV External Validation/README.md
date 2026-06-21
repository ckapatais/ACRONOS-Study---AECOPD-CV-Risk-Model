# MIMIC-IV External Validation

This folder contains the scripts used for external validation of the AECOPD-CV prediction model in the MIMIC-IV database.

The external validation framework was implemented using a frozen-model approach. Original regression coefficients and predefined median-imputation values derived from the development cohort were applied directly to the MIMIC-IV dataset without model refitting, recalibration, or coefficient updating. `predicted_probability_frozen` was generated using the original model coefficients, frozen development-cohort median imputation values, and prespecified internal calibration intercept/slope parameters derived before external validation. No external validation outcomes were used to estimate prediction parameters.

External validation analyses were performed using multiple laboratory extraction strategies:

- 6-hour laboratory window
- 12-hour laboratory window
- 24-hour laboratory window
- Anytime laboratory extraction window

For each extraction strategy, laboratory predictors were defined using the earliest available laboratory value within the corresponding time window following hospital admission.

## Validated Model Predictors

- Age
- History of heart failure
- History of atrial fibrillation
- Arterial pH
- Urea
- Lactate

## Composite Cardiovascular Outcome

- Myocardial infarction
- Pulmonary embolism
- Pulmonary edema / acute heart failure decompensation
- Acute arrhythmia

---

# Common Requirements

## Required Local Folder Structure

- Database.xlsx
- admissions.csv
- d_labitems.csv
- labevents.csv
- AECOPD_CV_FINAL_MODEL_Internal_Validation

The internal model directory must contain:

- model_coefficients_for_figures.csv
- recalibration_parameters.json
- analysis_dataset_internal_validation.csv

---

# MIMIC-IV External Validation: 6-Hour Laboratory Window

## Script Name

`mimiciv_external_validation_6h.py`

## Output Files

1. lab_itemids_used_6h.csv
2. patient_level_external_validation_predictions_MIMIC-IV_6h_labs.csv
3. Database_MIMIC-IV_external_validation_6h_labs_predictions.xlsx
4. lab_availability_6h.csv
5. outcome_definition_check_6h.csv
6. figure_roc_external_validation_6h.png
7. table_roc_external_validation_6h.csv
8. figure_decision_curve_external_validation_6h.png
9. table_decision_curve_external_validation_6h.csv
10. figure_calibration_external_validation_6h.png
11. table_calibration_external_validation_6h.csv
12. figure_quartiles_external_validation_6h.png
13. table_quartiles_external_validation_6h.csv
14. External_Validation_Composite_6h.png
15. summary_external_validation_6h.csv
16. summary_external_validation_6h.json
17. frozen_prediction_generation_metadata.json

---

# MIMIC-IV External Validation: 12-Hour Laboratory Window

## Script Name

`mimiciv_external_validation_12h.py`

## Output Files

1. lab_itemids_used_12h.csv
2. patient_level_external_validation_predictions_MIMIC-IV_12h_labs.csv
3. Database_MIMIC-IV_external_validation_12h_labs_predictions.xlsx
4. lab_availability_12h.csv
5. outcome_definition_check_12h.csv
6. figure_roc_external_validation_12h.png
7. table_roc_external_validation_12h.csv
8. figure_decision_curve_external_validation_12h.png
9. table_decision_curve_external_validation_12h.csv
10. figure_calibration_external_validation_12h.png
11. table_calibration_external_validation_12h.csv
12. figure_quartiles_external_validation_12h.png
13. table_quartiles_external_validation_12h.csv
14. External_Validation_Composite_12h.png
15. summary_external_validation_12h.csv
16. summary_external_validation_12h.json
17. frozen_prediction_generation_metadata.json

---

# MIMIC-IV External Validation: 24-Hour Laboratory Window

## Script Name

`mimiciv_external_validation_24h.py`

## Output Files

1. lab_itemids_used_24h.csv
2. patient_level_external_validation_predictions_MIMIC-IV_24h_labs.csv
3. Database_MIMIC-IV_external_validation_24h_labs_predictions.xlsx
4. lab_availability_24h.csv
5. outcome_definition_check_24h.csv
6. figure_roc_external_validation_24h.png
7. table_roc_external_validation_24h.csv
8. figure_decision_curve_external_validation_24h.png
9. table_decision_curve_external_validation_24h.csv
10. figure_calibration_external_validation_24h.png
11. table_calibration_external_validation_24h.csv
12. figure_quartiles_external_validation_24h.png
13. table_quartiles_external_validation_24h.csv
14. External_Validation_Composite_24h.png
15. summary_external_validation_24h.csv
16. summary_external_validation_24h.json
17. frozen_prediction_generation_metadata.json

---

# MIMIC-IV External Validation: Anytime Laboratory Extraction Window

## Script Name

`mimiciv_external_validation_anytime.py`

## Output Files

1. lab_itemids_used_anytime.csv
2. patient_level_external_validation_predictions_MIMIC-IV_anytime_labs.csv
3. Database_MIMIC-IV_external_validation_anytime_labs_predictions.xlsx
4. lab_availability_anytime.csv
5. outcome_definition_check_anytime.csv
6. figure_roc_external_validation_anytime.png
7. table_roc_external_validation_anytime.csv
8. figure_decision_curve_external_validation_anytime.png
9. table_decision_curve_external_validation_anytime.csv
10. figure_calibration_external_validation_anytime.png
11. table_calibration_external_validation_anytime.csv
12. figure_quartiles_external_validation_anytime.png
13. table_quartiles_external_validation_anytime.csv
14. External_Validation_Composite_anytime.png
15. summary_external_validation_anytime.csv
16. summary_external_validation_anytime.json
17. frozen_prediction_generation_metadata.json

---

# Notes

The scripts apply a frozen prediction framework. The model coefficients, median-imputation values, and frozen calibration parameters are taken from the internal model artifacts and are not estimated using the MIMIC-IV validation data.
