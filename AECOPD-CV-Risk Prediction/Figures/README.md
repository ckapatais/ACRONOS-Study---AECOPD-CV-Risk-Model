# FIGURES:
This folder contains all figures generated during the development, internal validation, external validation, and sensitivity analyses of the AECOPD-CV prediction model.

Figures were produced directly from the corresponding analysis scripts and were not manually modified after generation. The outputs include discrimination (ROC) analyses, calibration plots, decision curve analyses (DCA), risk-stratification visualizations, component-level sensitivity analyses, and supplementary methodological figures.

The figures were generated using the following analysis workflows:

* **Internal Validation** – Development cohort analyses, including model performance assessment, calibration evaluation, decision curve analysis, coefficient visualization, and risk-stratification analyses.
* **MIMIC-IV External Validation** – External validation of the frozen AECOPD-CV model using the MIMIC-IV database, including primary 6-hour laboratory extraction analyses and laboratory-availability sensitivity analyses.
* **eICU External Validation** – External validation of the frozen AECOPD-CV model using the eICU Collaborative Research Database, including 6-hour, 12-hour, 24-hour, and anytime laboratory extraction strategies.
* **Imputation Sensitivity Analyses** – Evaluation of model robustness under alternative missing-data handling strategies, including median, mean, K-nearest-neighbour, iterative, and complete-case approaches.

All figures were generated using reproducible scripted procedures implemented in Python. The corresponding scripts are available in the repository folders dedicated to internal validation, MIMIC-IV external validation, eICU external validation, and imputation sensitivity analyses.
