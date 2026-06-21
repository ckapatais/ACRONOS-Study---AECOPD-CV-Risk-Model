# Independent Temporal Pilot Implementation

This folder contains the scripts used for the independent temporal pilot implementation evaluation of the AECOPD-CV prediction model.

The pilot implementation cohort consisted of 33 consecutive patients evaluated through retrospective review of medical records. The purpose of this analysis was to assess model performance in an independent temporal cohort distinct from the development and external validation datasets.

The evaluation framework was implemented using a frozen-model approach. Original regression coefficients and predefined median-imputation values derived from the development cohort were applied directly without model refitting, recalibration, coefficient updating, or model redevelopment. No pilot implementation outcomes were used to estimate prediction parameters.

## Validated Model Predictors

- Age
- History of heart failure
- History of atrial fibrillation
- Arterial pH
- Urea
- Lactate

## Composite Cardiovascular Outcome

The composite cardiovascular outcome included:

- Myocardial infarction
- Pulmonary embolism
- Pulmonary edema
- Acute arrhythmia

## Cohort Summary

- Patients analysed: 33
- Composite cardiovascular events: 6
- Event rate: 18.2%
- AUC: 0.932 (95% CI 0.778–1.000)

---

# Independent Temporal Pilot Implementation Evaluation

## Script Name

`independent_temporal_pilot_implementation_visual_options.py`

## What the Script Does

The script:

- Loads the pilot implementation dataset.
- Reconstructs the composite cardiovascular outcome.
- Applies the frozen AECOPD-CV model coefficients.
- Generates patient-level predicted probabilities.
- Evaluates model discrimination.
- Produces risk-distribution visualizations.
- Generates publication-ready figures and summary tables.

No model refitting is performed.

## How to Run

```bash
cd "Your file directory"
python independent_temporal_pilot_implementation_visual_options.py
```

## Output Files

1. `independent_temporal_pilot_implementation_patient_predictions.csv`
   - Patient-level dataset containing predictors, outcomes, and predicted probabilities.

2. `independent_temporal_pilot_implementation_summary.csv`
   - Summary performance metrics including sample size, event rate, and discrimination.

3. `independent_temporal_pilot_implementation_metadata.json`
   - Metadata documenting model inputs, outcome definition, and analysis settings.

4. `01_roc_curve.png`
   - ROC curve with AUC and confidence interval.

5. `02_scatter_lowess.png`
   - Scatter plot of predicted probability versus observed outcome with LOWESS trend.

6. `03_violin_box_predicted_risk.png`
   - Violin and boxplot visualization of predicted risk stratified by outcome status.

7. `04_box_swarm_predicted_risk.png`
   - Boxplot and swarm plot of predicted risk stratified by outcome status.

8. `05_density_predicted_risk.png`
   - Kernel density distribution of predicted risk by outcome status.

9. `06_histogram_predicted_risk.png`
   - Histogram distribution of predicted probabilities by outcome status.

10. `independent_temporal_pilot_implementation_visual_options_composite.png`
    - Composite figure containing the ROC curve and risk-distribution visualizations.

---

# Notes

The script applies a frozen prediction framework. Original model coefficients and development-cohort imputation values are applied directly to the pilot implementation cohort. No model refitting, recalibration, coefficient updating, or outcome-informed parameter estimation is performed.

The final cardiovascular outcome is defined as the occurrence of at least one of:

- Myocardial infarction
- Pulmonary embolism
- Pulmonary edema / acute hf decompensation
- Acute arrhythmia
