# Titanic Survival Project

This project builds and documents a leakage-aware Titanic survival workflow centered on a learning notebook and a reusable Python pipeline.

## What This Repo Contains

1. A notebook-first workflow in `notebooks/data_processing.ipynb` for exploration, modeling, interpretation, and inference.
2. Reusable preprocessing and modeling code in `src/`.
3. A command-line pipeline that can train the final model bundle and create a Kaggle-ready submission.

## Setup

1. Create a virtual environment:
	`python -m venv venv`
2. Activate it on Windows:
	`venv\Scripts\activate`
3. Install dependencies:
	`pip install -r requirements.txt`

## Project Structure

- `data/raw/`: original Titanic CSV files.
- `data/processed/`: notebook-generated processed files used for study and comparison.
- `notebooks/data_processing.ipynb`: the main learning hub.
- `src/data_processing.py`: reusable preprocessing rules and feature transforms.
- `src/model.py`: training, prediction, evaluation, and coefficient helpers.
- `src/pipeline.py`: production-style script for training the final workflow and exporting outputs.
- `src/utils.py`: artifact and submission persistence helpers.
- `models/`: saved model bundles.
- `results/`: generated submissions and reusable outputs.

## Recommended Workflow

1. Use the notebook to understand the data, validate modeling decisions, and inspect results.
2. Once the workflow is chosen, run the reusable pipeline script to create the final artifacts.
3. Submit the generated CSV to Kaggle.

## Run the Final Pipeline

Train the final workflow and generate a submission:

```bash
python -m src.pipeline --workflow experiment_1
```

This creates:

1. A saved model bundle at `models/titanic_logistic_bundle.pkl`
2. A submission file at `results/submission.csv`

You can also override paths:

```bash
python -m src.pipeline --workflow experiment_1 --submission-path notebooks/submission.csv
```

## How to Use the Saved Bundle

The model bundle stores:

1. The final workflow choice.
2. Learned preprocessing rules.
3. The aligned feature schema.
4. The trained logistic regression model.

This is enough to support repeatable inference on Titanic-format test data.

### Make a Kaggle Submission From an Existing Bundle

If you have already trained the project once and only want to recreate predictions from the saved pickle bundle:

```bash
python -m src.pipeline --mode predict_submission --bundle-path models/titanic_logistic_bundle.pkl --input-path data/raw/test.csv --output-path results/submission_from_bundle.csv
```

Use this when the input file has the same raw Titanic-style schema as the original Kaggle test set and you want a two-column submission output.

### Score a New Raw Data Source

If you want to use the trained pickle file on a new raw CSV source, run:

```bash
python -m src.pipeline --mode score_data --bundle-path models/titanic_logistic_bundle.pkl --input-path data/raw/test.csv --output-path results/scored_test.csv --id-column PassengerId
```

This creates a scored table with:

1. An identifier column.
2. A predicted class column.
3. A predicted probability column.

Important requirement:
The new source must still contain the raw columns expected by the preprocessing pipeline, such as `Pclass`, `Sex`, `Age`, `Fare`, `Embarked`, `SibSp`, `Parch`, `Name`, `Ticket`, and `Cabin`.

If your new source uses different column names or a different schema, you must map it into the Titanic raw schema before scoring.

### What the Pickle File Actually Does

The pickle file is not just trained coefficients. It is the full inference contract for the project:

1. It remembers which workflow won.
2. It stores the preprocessing rules learned from training data.
3. It stores the exact encoded feature schema expected by the model.
4. It stores the trained logistic regression object.

That is why loading only coefficients would be insufficient. The model also needs the same transformations and the same final column layout.

### How To Adapt a Truly New Source

For a new source outside Kaggle, follow this sequence:

1. Start with raw passenger-level data.
2. Rename or map columns so they match the Titanic raw schema expected by the bundle.
3. Save that mapped dataset as CSV.
4. Run `score_data` mode with the saved bundle.
5. Inspect both predicted class and predicted probability.

If the source does not represent Titanic-style passengers, then this trained model is not appropriate and should not be reused without retraining.

## Kaggle Next Steps

1. Run the notebook end to end and confirm the chosen workflow is still the best validated option.
2. Run `python -m src.pipeline --workflow experiment_1`.
3. Inspect `results/submission.csv`.
4. Upload the CSV to the Kaggle Titanic competition.
5. Record your public leaderboard score.
6. Create one controlled new experiment at a time and resubmit.

## Deployment Notes

For this project, deployment should be understood as batch inference rather than a live web service.

The clean deployment pattern is:

1. Save the trained model bundle.
2. Load the bundle for future inference on new Titanic-format data.
3. Reproduce the exact preprocessing and schema alignment before prediction.

If you later want to build an API, the bundle produced by `src.pipeline` is the correct starting artifact.