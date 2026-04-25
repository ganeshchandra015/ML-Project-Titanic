import argparse

import pandas as pd

from src import config
from src.data_processing import (
	VALID_WORKFLOWS,
	align_to_reference,
	build_preprocessing_rules,
	one_hot_encode,
	transform_features,
)
from src.model import predict_labels_and_probabilities, train_logistic_regression
from src.utils import build_submission_frame, load_pickle, save_pickle, save_submission


def _encode_features_for_inference(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
	transformed = transform_features(frame, bundle["rules"])
	encoded = one_hot_encode(transformed)
	return align_to_reference(encoded, pd.Index(bundle["feature_columns"]))


def train_final_pipeline(
	workflow: str = "experiment_1",
	train_path: str = config.TRAIN_DATA_PATH,
	test_path: str = config.TEST_DATA_PATH,
	model_bundle_path: str = config.DEFAULT_MODEL_BUNDLE_PATH,
	submission_path: str = config.DEFAULT_SUBMISSION_PATH,
) -> dict:
	if workflow not in VALID_WORKFLOWS:
		raise ValueError(f"workflow must be one of {sorted(VALID_WORKFLOWS)}")

	train_frame = pd.read_csv(train_path)
	test_frame = pd.read_csv(test_path)

	features = train_frame.drop(columns=["Survived"])
	target = train_frame["Survived"]
	test_ids = test_frame["PassengerId"].copy()

	rules = build_preprocessing_rules(features, workflow=workflow)
	train_transformed = transform_features(features, rules)
	test_transformed = transform_features(test_frame, rules)

	train_encoded = one_hot_encode(train_transformed)
	test_encoded = one_hot_encode(test_transformed)
	test_encoded = align_to_reference(test_encoded, train_encoded.columns)

	model = train_logistic_regression(train_encoded, target)
	predictions, probabilities = predict_labels_and_probabilities(model, test_encoded)

	bundle = {
		"workflow": workflow,
		"rules": rules,
		"feature_columns": list(train_encoded.columns),
		"model": model,
	}
	save_pickle(bundle, model_bundle_path)

	submission = build_submission_frame(test_ids, predictions)
	save_submission(submission, submission_path)

	return {
		"bundle": bundle,
		"submission": submission,
		"probabilities": probabilities,
		"train_shape": train_encoded.shape,
		"test_shape": test_encoded.shape,
		"submission_path": submission_path,
		"model_bundle_path": model_bundle_path,
	}


def predict_with_saved_bundle(bundle_path: str, test_path: str, submission_path: str) -> pd.DataFrame:
	bundle = load_pickle(bundle_path)
	test_frame = pd.read_csv(test_path)
	test_ids = test_frame["PassengerId"].copy()

	test_encoded = _encode_features_for_inference(test_frame, bundle)
	predictions, _ = predict_labels_and_probabilities(bundle["model"], test_encoded)

	submission = build_submission_frame(test_ids, predictions)
	save_submission(submission, submission_path)
	return submission


def score_new_data_with_saved_bundle(
	bundle_path: str,
	input_path: str,
	output_path: str | None = None,
	id_column: str = "PassengerId",
	prediction_column: str = "Predicted_Survived",
	probability_column: str = "Predicted_Probability",
) -> pd.DataFrame:
	bundle = load_pickle(bundle_path)
	input_frame = pd.read_csv(input_path)
	encoded = _encode_features_for_inference(input_frame, bundle)
	predictions, probabilities = predict_labels_and_probabilities(bundle["model"], encoded)

	if id_column in input_frame.columns:
		identifier_values = input_frame[id_column].copy()
		identifier_name = id_column
	else:
		identifier_values = pd.Series(range(1, len(input_frame) + 1), name="RowId")
		identifier_name = "RowId"

	results = pd.DataFrame(
		{
			identifier_name: identifier_values,
			prediction_column: predictions,
			probability_column: probabilities,
		}
	)

	if output_path:
		save_submission(results, output_path)

	return results


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train the Titanic pipeline or reuse a saved bundle for inference.")
	parser.add_argument(
		"--mode",
		default="train",
		choices=["train", "predict_submission", "score_data"],
		help="train: fit final model bundle, predict_submission: make a Kaggle-style submission from a saved bundle, score_data: score new raw data with predictions and probabilities.",
	)
	parser.add_argument("--workflow", default="experiment_1", choices=sorted(VALID_WORKFLOWS))
	parser.add_argument("--train-path", default=config.TRAIN_DATA_PATH)
	parser.add_argument("--test-path", default=config.TEST_DATA_PATH)
	parser.add_argument("--model-bundle-path", default=config.DEFAULT_MODEL_BUNDLE_PATH)
	parser.add_argument("--submission-path", default=config.DEFAULT_SUBMISSION_PATH)
	parser.add_argument("--bundle-path", default=config.DEFAULT_MODEL_BUNDLE_PATH)
	parser.add_argument("--input-path", default=config.TEST_DATA_PATH)
	parser.add_argument("--output-path", default=None)
	parser.add_argument("--id-column", default="PassengerId")
	parser.add_argument("--prediction-column", default="Predicted_Survived")
	parser.add_argument("--probability-column", default="Predicted_Probability")
	return parser


def main() -> None:
	parser = build_argument_parser()
	args = parser.parse_args()

	if args.mode == "train":
		artifacts = train_final_pipeline(
			workflow=args.workflow,
			train_path=args.train_path,
			test_path=args.test_path,
			model_bundle_path=args.model_bundle_path,
			submission_path=args.submission_path,
		)
		print(f"Workflow: {artifacts['bundle']['workflow']}")
		print(f"Encoded train shape: {artifacts['train_shape']}")
		print(f"Encoded test shape: {artifacts['test_shape']}")
		print(f"Saved model bundle to: {artifacts['model_bundle_path']}")
		print(f"Saved submission to: {artifacts['submission_path']}")
		print(artifacts["submission"].head())
		return

	if args.mode == "predict_submission":
		submission_path = args.output_path or args.submission_path
		submission = predict_with_saved_bundle(
			bundle_path=args.bundle_path,
			test_path=args.input_path,
			submission_path=submission_path,
		)
		print(f"Saved submission to: {submission_path}")
		print(submission.head())
		return

	results = score_new_data_with_saved_bundle(
		bundle_path=args.bundle_path,
		input_path=args.input_path,
		output_path=args.output_path,
		id_column=args.id_column,
		prediction_column=args.prediction_column,
		probability_column=args.probability_column,
	)
	if args.output_path:
		print(f"Saved scored data to: {args.output_path}")
	print(results.head())


if __name__ == "__main__":
	main()