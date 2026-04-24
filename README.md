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