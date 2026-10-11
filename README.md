# Vehicle Insurance Cross-Sell Response Predictor

A Streamlit decision-support app that estimates customer response to a vehicle-insurance cross-sell campaign. It scores an individual customer and summarizes predictions for the supplied test cohort.

## What It Does

- Estimates an individual customer's response probability from nine customer and policy fields.
- Compares that score with a validation-selected campaign threshold.
- Summarizes the model's scores for 127,037 unlabeled test customers.
- Includes an exploratory analysis and model experiments in the Jupyter notebook.

## Model

The app loads `insurance_response_model_no_gender.pkl`, a scikit-learn pipeline with categorical preprocessing and a `HistGradientBoostingClassifier`. Gender is not an input to this app model. The saved artifact is version-sensitive, so use the Python and package versions pinned in `requirements.txt`.

The operating threshold is approximately **0.2249**. It was selected on a validation split to balance precision and recall for responders; it is not a universal business threshold.

## Holdout Results

The app model was evaluated on a stratified holdout split from `train.csv`: 70% training, 15% validation, and 15% holdout, using `random_state=42`. The threshold was selected using validation data and then evaluated on the untouched holdout.

| Metric | Holdout result |
| --- | ---: |
| Accuracy | 77.91% |
| ROC-AUC | 0.8580 |
| PR-AUC | 0.3720 |
| Precision for responders | 32.79% |
| Recall for responders | 76.44% |
| F1 for responders | 0.4589 |

Responders make up about 12.3% of the labeled data, so predicting "no response" for everyone would yield about 87.7% accuracy. The app's lower accuracy reflects its threshold's emphasis on identifying more responders. On the unlabeled test cohort, the mean predicted probability is **12.3%** and **28.7%** of customers are above threshold. These are model estimates, not observed response outcomes.

## Run Locally

Requires 64-bit Python 3.13. In PowerShell, from the project directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Deploy

To showcase the app, push the project to GitHub and deploy it with Streamlit Community Cloud. Select `app.py` as the main file. Keep the model artifact and `final_test_predictions.csv` in the repository root because the app loads both relative to `app.py`.

## Project Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit application |
| `insurance_response_model_no_gender.pkl` | Saved model used by the app |
| `final_test_predictions.csv` | Model scores for the unlabeled test cohort |
| `stat.ipynb` | Exploratory analysis and model experiments |
| `requirements.txt` | Pinned app and notebook dependencies |

## Limitations And Publishing Notes

- The raw `train.csv`, `test.csv`, and `sample_submission.csv` files are not included in this repository. The matching source is the [Kaggle Health Insurance Cross Sell Prediction competition](https://www.kaggle.com/competitions/health-insurance-cross-sell-prediction); obtain the data there and follow its rules.
- `test.csv` has no response labels; test-set accuracy cannot be measured from the files in this project.
- Holdout results come from one split and do not guarantee future campaign performance.
- A score above threshold is not certainty that a customer will respond. The holdout precision is 32.79%.
- The notebook contains saved outputs but has not been validated by running every cell from a clean kernel.
- The source link does not grant permission to redistribute the data. Review the competition's current terms before reusing or sharing any source files or derived artifacts.
- MIT is the intended license for the original project code. Add a `LICENSE` file with the copyright holder's name before publishing; it does not grant rights to Kaggle data or other third-party artifacts.