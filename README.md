# Diabetes Progression Data Science Final Project

Repository: https://github.com/kanezuki/CS439-Final-Project

This folder contains the reproducible code and output files for the final project. The written report is kept separately in `report_submission/final_report.pdf` in the submission zip.

## Reproducing the analysis

From the project root:

```bash
pip install -r requirements.txt
python src/run_analysis.py
```

The script exports:

- `data/diabetes_dataset.csv`
- model metrics in `outputs/*.csv`
- generated figures in `outputs/figures/`
- summary metadata in `outputs/metadata.json`

## Main result summary

Regression: the best model by RMSE was the random forest regressor.

Classification: the best model by ROC-AUC was logistic regression.

The report uses the generated values from the included code, not placeholder values.
