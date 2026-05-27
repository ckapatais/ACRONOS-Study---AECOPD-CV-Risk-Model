eICU External Validation
This folder contains the scripts used for external validation of the AECOPD-CV prediction model in the eICU Collaborative Research Database.

The external validation framework was implemented using a frozen-model approach. Original regression coefficients and predefined median-imputation values derived from the development cohort were applied directly to the eICU dataset without model refitting, recalibration, or coefficient updating. External validation analyses were performed across four predefined laboratory extraction strategies:
- 6-hour laboratory window (0–360 minutes).
- 12-hour laboratory window (0–720 minutes).
- 24-hour laboratory window (0–1440 minutes).
- Anytime laboratory extraction window (ICU admission until ICU discharge).

For each extraction strategy, laboratory predictors were defined using the earliest available laboratory value within the corresponding time window following ICU admission. Predictor extraction, cohort construction, outcome generation, prediction generation, and performance evaluation were implemented programmatically using reproducible scripted workflows. The validated model included the following predefined predictors:
1. Age.
2. History of heart failure.
3. History of atrial fibrillation.
4. Arterial pH.
5. Urea.
6. Lactate.

Because urea measurements are not consistently available in eICU, blood urea nitrogen (BUN) values were converted using the standard clinical relationship:
Urea=BUN×2.14
Direct urea measurements, when available, were used only as fallback values in the absence of BUN data.

The composite cardiovascular outcome included:
- Myocardial infarction.
- Pulmonary embolism.
- Pulmonary edema.
- Acute arrhythmia.

All scripts were developed to ensure methodological reproducibility and consistency across temporal sensitivity analyses.
