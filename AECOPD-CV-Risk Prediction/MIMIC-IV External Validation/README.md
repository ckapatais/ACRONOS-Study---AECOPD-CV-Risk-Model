MIMIC-IV External Validation

This folder contains the scripts used for external validation of the AECOPD-CV prediction model in the MIMIC-IV database.

The external validation framework was implemented using a frozen-model approach. Original regression coefficients and predefined median-imputation values derived from the development cohort were applied directly to the MIMIC-IV dataset without model refitting, recalibration, or coefficient updating. Predicted_probability_frozen was generated using the original model coefficients, frozen development-cohort median imputation values, and prespecified internal calibration intercept/slope parameters derived before external validation. No external validation outcomes were used to estimate prediction parameters.

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

MIMIC-IV External Validation: 6-hour Laboratory Window

This script performs the MIMIC-IV external validation of the AECOPD-CV model using a 6-hour laboratory extraction window.

Script name: mimiciv_external_validation_6h.py
What the script does: The script reproduces the 6-hour MIMIC-IV external validation workflow. It:

Loads the MIMIC-IV external validation cohort from Database.xlsx.
Uses the sheet AECOPD_validation_dataset_v3.
Applies the final composite cardiovascular outcome definition using:
mi_inclusive_event
arrhythmia_inclusive_event
pulmonary edema / hf_decompensation_event
pe_inclusive_event
Extracts the earliest plausible pH, BUN/urea, and lactate values within 6 hours from admission using MIMIC-IV labevents.
Converts BUN to urea using a conversion factor of 2.14.
Applies predefined plausibility filters for laboratory values.
Applies the frozen AECOPD-CV model coefficients.
Applies the frozen internal calibration intercept and slope.
Generates patient-level predicted probabilities.
Evaluates discrimination, calibration, clinical utility, and risk stratification.

No model refitting is performed in the MIMIC-IV validation cohort.

Required local folder structure
The script expects the following files and folders:
1. Database.xlsx (obtained from mimiciv_cohort_derivation.py script).
2. admissions.csv (obtained from MIMIC-IV once granted access from PhysioNet).
3. d_labitems.csv (obtained from MIMIC-IV once granted access from PhysioNet).
4. labevents.csv (obtained from MIMIC-IV once granted access from PhysioNet).
5. AECOPD_CV_FINAL_MODEL_Internal Validation (obtained from internal_validation_aecopd_cv_model.py script). The internal model directory must contain:
5a. model_coefficients_for_figures.csv
5b. recalibration_parameters.json
5c. analysis_dataset_internal_validation.csv
The aforementioned files are generated automatically once you run the script named as: internal_validation_aecopd_cv_model.py

How to run
Open Command Prompt and run:
cd "Your file directory"
python mimiciv_external_validation_6h.py

Output folder:
The script creates outputs in:

[Your directory]\external_validation_6h_original

The script generates 17 output files:
1. lab_itemids_used_6h.csv (Laboratory item IDs used to identify pH, BUN/urea, and lactate).
2. patient_level_external_validation_predictions_MIMIC-IV_6h_labs.csv (Patient-level dataset with extracted 6-hour laboratory values and predicted probabilities).
3. Database_MIMIC-IV_external_validation_6h_labs_predictions.xlsx (Excel version of the patient-level prediction dataset, laboratory availability table, and laboratory item ID table).
4. lab_availability_6h.csv (Availability and missingness summary for pH, urea, lactate, and complete laboratory availability).
5. outcome_definition_check_6h.csv (Counts for each cardiovascular outcome component and the final reconstructed composite outcome).
6. figure_roc_external_validation_6h.png (ROC curve for the 6-hour MIMIC-IV external validation).
7. table_roc_external_validation_6h.csv (AUC and bootstrap-derived 95% confidence interval).
8. figure_decision_curve_external_validation_6h.png (Decision curve analysis figure).
9. table_decision_curve_external_validation_6h.csv (Net benefit values across threshold probabilities).
10. figure_calibration_external_validation_6h.png (LOWESS-smoothed calibration plot).
11. table_calibration_external_validation_6h.csv (Calibration intercept, calibration slope, and plotting range).
12. figure_quartiles_external_validation_6h.png (Observed event rates across predicted-risk quartiles).
13. table_quartiles_external_validation_6h.csv (Quartile-level predicted risk, observed event rate, and 95% confidence intervals).
14. External_Validation_Composite_6h.png (Combined figure containing ROC, decision curve, calibration, and quartile plots).
15. summary_external_validation_6h.csv (Main summary table with sample size, event count, event rate, AUC, Brier score, calibration intercept, and calibration slope).
16. summary_external_validation_6h.json (JSON version of the external validation summary).
17. frozen_prediction_generation_metadata.json (Metadata documenting coefficient sources, frozen calibration parameters, imputation medians, and model definition).

Main expected results
The script reports:
Rows analysed: 716

Events preserved from final outcome: 269
The final model performance is written to:
summary_external_validation_6h.csv
summary_external_validation_6h.json
Notes

The script applies a frozen prediction framework. The model coefficients, median-imputation values, and frozen calibration parameters are taken from the internal model artifacts and are not estimated using the MIMIC-IV validation data.

The final cardiovascular outcome is defined as the occurrence of at least one of:
myocardial infarction
pulmonary embolism
pulmonary edema / acute heart failure decompensation
acute arrhythmia
