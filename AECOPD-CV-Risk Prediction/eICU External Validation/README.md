# eICU External Validation

This folder contains the scripts used for external validation of the AECOPD-CV prediction model in the eICU Collaborative Research Database.

The external validation framework was implemented using a frozen-model approach. Original regression coefficients and predefined median-imputation values derived from the development cohort were applied directly to the eICU dataset without model refitting, recalibration, coefficient updating, or model redevelopment. Predicted probabilities were generated using the original model coefficients and frozen development-cohort median-imputation values. No external validation outcomes were used to estimate prediction parameters.

External validation analyses were performed across four predefined laboratory extraction strategies:

- 6-hour laboratory window (0–360 minutes)
- 12-hour laboratory window (0–720 minutes)
- 24-hour laboratory window (0–1440 minutes)
- Anytime laboratory extraction window (ICU admission until ICU discharge)

For each extraction strategy, laboratory predictors were defined using the earliest available laboratory value within the corresponding time window following ICU admission.

Across all four eICU external validation scripts, the final analysed cohort included 4210 patients and 1586 composite cardiovascular events.

## Validated Model Predictors

- Age
- History of heart failure
- History of atrial fibrillation
- Arterial pH
- Urea
- Lactate

Because urea measurements are not consistently available in eICU, blood urea nitrogen (BUN) values were converted using the standard clinical relationship:

```text
Urea = BUN × 2.14
```

Direct urea measurements, when available, were used only as fallback values in the absence of BUN data.

## Composite Cardiovascular Outcome

The composite cardiovascular outcome included:

- Myocardial infarction
- Pulmonary embolism
- Pulmonary edema
- Acute arrhythmia

All scripts were developed using fully reproducible computational workflows to ensure consistency across cohort construction, predictor extraction, outcome generation, prediction generation, and performance evaluation procedures.

---

# Common Requirements

## Required eICU Files

The scripts require access to the following eICU tables:

- `patient.csv` or `patient.csv.gz`
- `diagnosis.csv` or `diagnosis.csv.gz`
- `lab.csv` or `lab.csv.gz`
- `pastHistory.csv` or `pastHistory.csv.gz`

The scripts additionally allow optional external model artifact files:

- `model_coefficients_for_figures.csv`
- A medians CSV file containing predefined imputation medians

If no external coefficient or median file is provided, the embedded frozen coefficients and embedded development-cohort medians are used automatically.

## Required Python Packages

The scripts require:

- `numpy`
- `pandas`
- `matplotlib`
- `statsmodels`
- `scikit-learn`
- `Pillow`

---

# eICU External Validation: 6-Hour Laboratory Window

## Script Name

`eicu_external_validation_6h.py`

## What the Script Does

The script reproduces the 6-hour eICU external validation workflow. It:

- Loads the required eICU tables.
- Identifies patients with acute exacerbation of COPD using diagnosis records.
- Retains the first ICU stay per patient when multiple ICU stays are present.
- Extracts history of heart failure and history of atrial fibrillation from past medical history records.
- Reconstructs the composite cardiovascular outcome using diagnosis records.
- Extracts the earliest plausible pH, BUN/urea, and lactate values within 0–360 minutes from ICU admission.
- Converts BUN to urea using a conversion factor of 2.14.
- Uses direct urea only as a fallback when BUN is unavailable.
- Applies predefined plausibility filters and clipping rules for laboratory predictors.
- Applies frozen AECOPD-CV model coefficients.
- Applies frozen development-cohort median imputation values.
- Generates patient-level predicted probabilities.
- Evaluates discrimination, calibration, clinical utility, and risk stratification.
- Generates publication-ready figures and tables.

No model refitting is performed in the eICU validation cohort.

## How to Run

```bash
cd "Your file directory"
python eicu_external_validation_6h.py --eicu-dir "path_to_eicu_folder" --output "external_validation_6h"
```

## Laboratory Window

0–360 minutes following ICU admission.

## Output Folder

`external_validation_6h`

## Output Files

The script generates the following output files:

1. `patient_level_predictions.csv`  
   Patient-level prediction dataset containing cohort characteristics, extracted predictors, outcome variables, and model predictions.

2. `patient_level_best_model_figure_inputs.csv`  
   Patient-level dataset used to generate the publication figures for the selected AECOPD-CV model.

3. `model_comparison_results.csv`  
   Model comparison table retained for audit and reproducibility.

4. `summary_best_model_figures.json`  
   JSON summary of the selected model used in the publication figures.

5. `summary_eicu_6h.json`  
   Main JSON summary of the 6-hour eICU external validation analysis.

6. `table_lab_availability_6h.csv`  
   Laboratory availability table for the 6-hour extraction window.

7. `figure_roc_external_validation.png`  
   ROC curve for the eICU external validation analysis.

8. `table_roc_external_validation.csv`  
   AUC and bootstrap-derived 95% confidence interval.

9. `figure_decision_curve_external_validation.png`  
   Decision curve analysis figure.

10. `table_decision_curve_external_validation.csv`  
    Net benefit values across threshold probabilities.

11. `figure_calibration_external_validation.png`  
    LOWESS-smoothed calibration plot.

12. `table_calibration_external_validation.csv`  
    Calibration intercept, calibration slope, and plotting range.

13. `figure_quartiles_external_validation.png`  
    Observed event rates across predicted-risk quartiles.

14. `table_quartiles_external_validation.csv`  
    Quartile-level predicted risk, observed event rate, and confidence intervals.

15. `External_Validation_Composite_6h.png`  
    Combined figure containing ROC, decision curve, calibration, and quartile plots.

---

# eICU External Validation: 12-Hour Laboratory Window

## Script Name

`eicu_external_validation_12h.py`

## What the Script Does

The script reproduces the 12-hour eICU external validation workflow. It:

- Loads the required eICU tables.
- Identifies patients with acute exacerbation of COPD using diagnosis records.
- Retains the first ICU stay per patient when multiple ICU stays are present.
- Extracts history of heart failure and history of atrial fibrillation from past medical history records.
- Reconstructs the composite cardiovascular outcome using diagnosis records.
- Extracts the earliest plausible pH, BUN/urea, and lactate values within 0–720 minutes from ICU admission.
- Converts BUN to urea using a conversion factor of 2.14.
- Uses direct urea only as a fallback when BUN is unavailable.
- Applies predefined plausibility filters and clipping rules for laboratory predictors.
- Applies frozen AECOPD-CV model coefficients.
- Applies frozen development-cohort median imputation values.
- Generates patient-level predicted probabilities.
- Evaluates discrimination, calibration, clinical utility, and risk stratification.
- Generates publication-ready figures and tables.

No model refitting is performed in the eICU validation cohort.

## How to Run

```bash
cd "Your file directory"
python eicu_external_validation_12h.py --eicu-dir "path_to_eicu_folder" --output "external_validation_12h"
```

## Laboratory Window

0–720 minutes following ICU admission.

## Output Folder

`external_validation_12h`

## Output Files

The script generates the following output files:

1. `patient_level_predictions.csv`  
   Patient-level prediction dataset containing cohort characteristics, extracted predictors, outcome variables, and model predictions.

2. `patient_level_best_model_figure_inputs.csv`  
   Patient-level dataset used to generate the publication figures for the selected AECOPD-CV model.

3. `model_comparison_results.csv`  
   Model comparison table retained for audit and reproducibility.

4. `summary_best_model_figures.json`  
   JSON summary of the selected model used in the publication figures.

5. `summary_eicu_12h.json`  
   Main JSON summary of the 12-hour eICU external validation analysis.

6. `table_lab_availability_12h.csv`  
   Laboratory availability table for the 12-hour extraction window.

7. `figure_roc_external_validation.png`  
   ROC curve for the eICU external validation analysis.

8. `table_roc_external_validation.csv`  
   AUC and bootstrap-derived 95% confidence interval.

9. `figure_decision_curve_external_validation.png`  
   Decision curve analysis figure.

10. `table_decision_curve_external_validation.csv`  
    Net benefit values across threshold probabilities.

11. `figure_calibration_external_validation.png`  
    LOWESS-smoothed calibration plot.

12. `table_calibration_external_validation.csv`  
    Calibration intercept, calibration slope, and plotting range.

13. `figure_quartiles_external_validation.png`  
    Observed event rates across predicted-risk quartiles.

14. `table_quartiles_external_validation.csv`  
    Quartile-level predicted risk, observed event rate, and confidence intervals.

15. `External_Validation_Composite_12h.png`  
    Combined figure containing ROC, decision curve, calibration, and quartile plots.

---

# eICU External Validation: 24-Hour Laboratory Window

## Script Name

`eicu_external_validation_24h.py`

## What the Script Does

The script reproduces the 24-hour eICU external validation workflow. It:

- Loads the required eICU tables.
- Identifies patients with acute exacerbation of COPD using diagnosis records.
- Retains the first ICU stay per patient when multiple ICU stays are present.
- Extracts history of heart failure and history of atrial fibrillation from past medical history records.
- Reconstructs the composite cardiovascular outcome using diagnosis records.
- Extracts the earliest plausible pH, BUN/urea, and lactate values within 0–1440 minutes from ICU admission.
- Converts BUN to urea using a conversion factor of 2.14.
- Uses direct urea only as a fallback when BUN is unavailable.
- Applies predefined plausibility filters and clipping rules for laboratory predictors.
- Applies frozen AECOPD-CV model coefficients.
- Applies frozen development-cohort median imputation values.
- Generates patient-level predicted probabilities.
- Evaluates discrimination, calibration, clinical utility, and risk stratification.
- Generates publication-ready figures and tables.

No model refitting is performed in the eICU validation cohort.

## How to Run

```bash
cd "Your file directory"
python eicu_external_validation_24h.py --eicu-dir "path_to_eicu_folder" --output "external_validation_24h"
```

## Laboratory Window

0–1440 minutes following ICU admission.

## Output Folder

`external_validation_24h`

## Output Files

The script generates the following output files:

1. `patient_level_predictions.csv`  
   Patient-level prediction dataset containing cohort characteristics, extracted predictors, outcome variables, and model predictions.

2. `patient_level_best_model_figure_inputs.csv`  
   Patient-level dataset used to generate the publication figures for the selected AECOPD-CV model.

3. `model_comparison_results.csv`  
   Model comparison table retained for audit and reproducibility.

4. `summary_best_model_figures.json`  
   JSON summary of the selected model used in the publication figures.

5. `summary_eicu_24h.json`  
   Main JSON summary of the 24-hour eICU external validation analysis.

6. `table_lab_availability_24h.csv`  
   Laboratory availability table for the 24-hour extraction window.

7. `figure_roc_external_validation.png`  
   ROC curve for the eICU external validation analysis.

8. `table_roc_external_validation.csv`  
   AUC and bootstrap-derived 95% confidence interval.

9. `figure_decision_curve_external_validation.png`  
   Decision curve analysis figure.

10. `table_decision_curve_external_validation.csv`  
    Net benefit values across threshold probabilities.

11. `figure_calibration_external_validation.png`  
    LOWESS-smoothed calibration plot.

12. `table_calibration_external_validation.csv`  
    Calibration intercept, calibration slope, and plotting range.

13. `figure_quartiles_external_validation.png`  
    Observed event rates across predicted-risk quartiles.

14. `table_quartiles_external_validation.csv`  
    Quartile-level predicted risk, observed event rate, and confidence intervals.

15. `External_Validation_Composite_24h.png`  
    Combined figure containing ROC, decision curve, calibration, and quartile plots.

---

# eICU External Validation: Anytime Laboratory Extraction Window

## Script Name

`eicu_external_validation_anytime.py`

## What the Script Does

The script reproduces the anytime eICU external validation workflow. It:

- Loads the required eICU tables.
- Identifies patients with acute exacerbation of COPD using diagnosis records.
- Retains the first ICU stay per patient when multiple ICU stays are present.
- Extracts history of heart failure and history of atrial fibrillation from past medical history records.
- Reconstructs the composite cardiovascular outcome using diagnosis records.
- Extracts the earliest plausible pH, BUN/urea, and lactate values at any time after ICU admission and before ICU discharge.
- Converts BUN to urea using a conversion factor of 2.14.
- Uses direct urea only as a fallback when BUN is unavailable.
- Applies predefined plausibility filters and clipping rules for laboratory predictors.
- Applies frozen AECOPD-CV model coefficients.
- Applies frozen development-cohort median imputation values.
- Generates patient-level predicted probabilities.
- Evaluates discrimination, calibration, clinical utility, and risk stratification.
- Generates publication-ready figures and tables.

No model refitting is performed in the eICU validation cohort.

## How to Run

```bash
cd "Your file directory"
python eicu_external_validation_anytime.py --eicu-dir "path_to_eicu_folder" --output "external_validation_anytime"
```

## Laboratory Window

ICU admission until ICU discharge.

## Output Folder

`external_validation_anytime`

## Output Files

The script generates the following output files:

1. `patient_level_predictions.csv`  
   Patient-level prediction dataset containing cohort characteristics, extracted predictors, outcome variables, and model predictions.

2. `patient_level_best_model_figure_inputs.csv`  
   Patient-level dataset used to generate the publication figures for the selected AECOPD-CV model.

3. `model_comparison_results.csv`  
   Model comparison table retained for audit and reproducibility.

4. `summary_best_model_figures.json`  
   JSON summary of the selected model used in the publication figures.

5. `summary_eicu_anytime.json`  
   Main JSON summary of the anytime eICU external validation analysis.

6. `table_lab_availability_anytime.csv`  
   Laboratory availability table for the anytime extraction window.

7. `figure_roc_external_validation.png`  
   ROC curve for the eICU external validation analysis.

8. `table_roc_external_validation.csv`  
   AUC and bootstrap-derived 95% confidence interval.

9. `figure_decision_curve_external_validation.png`  
   Decision curve analysis figure.

10. `table_decision_curve_external_validation.csv`  
    Net benefit values across threshold probabilities.

11. `figure_calibration_external_validation.png`  
    LOWESS-smoothed calibration plot.

12. `table_calibration_external_validation.csv`  
    Calibration intercept, calibration slope, and plotting range.

13. `figure_quartiles_external_validation.png`  
    Observed event rates across predicted-risk quartiles.

14. `table_quartiles_external_validation.csv`  
    Quartile-level predicted risk, observed event rate, and confidence intervals.

15. `External_Validation_Composite_anytime.png`  
    Combined figure containing ROC, decision curve, calibration, and quartile plots.

---

# Notes

The scripts apply a frozen prediction framework. Original model coefficients and development-cohort median-imputation values are applied directly to the eICU cohort. No model refitting, recalibration, coefficient updating, model redevelopment, or outcome-informed parameter estimation is performed during external validation.

The final cardiovascular outcome is defined as the occurrence of at least one of:

- Myocardial infarction
- Pulmonary embolism
- Pulmonary edema
- Acute arrhythmia

Publication figures display the selected AECOPD-CV model. Additional candidate model outputs retained in `patient_level_predictions.csv` and `model_comparison_results.csv` are preserved for audit and reproducibility purposes.
