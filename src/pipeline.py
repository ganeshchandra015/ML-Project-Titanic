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
from src.utils import build_submission_frame, save_pickle, save_submission


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
	bundle = pd.read_pickle(bundle_path)
	test_frame = pd.read_csv(test_path)
	test_ids = test_frame["PassengerId"].copy()

	test_transformed = transform_features(test_frame, bundle["rules"])
	test_encoded = one_hot_encode(test_transformed)
	test_encoded = align_to_reference(test_encoded, pd.Index(bundle["feature_columns"]))
	predictions, _ = predict_labels_and_probabilities(bundle["model"], test_encoded)

	submission = build_submission_frame(test_ids, predictions)
	save_submission(submission, submission_path)
	return submission


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train the final Titanic pipeline and create a Kaggle submission.")
	parser.add_argument("--workflow", default="experiment_1", choices=sorted(VALID_WORKFLOWS))
	parser.add_argument("--train-path", default=config.TRAIN_DATA_PATH)
	parser.add_argument("--test-path", default=config.TEST_DATA_PATH)
	parser.add_argument("--model-bundle-path", default=config.DEFAULT_MODEL_BUNDLE_PATH)
	parser.add_argument("--submission-path", default=config.DEFAULT_SUBMISSION_PATH)
	return parser


def main() -> None:
	parser = build_argument_parser()
	args = parser.parse_args()
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


if __name__ == "__main__":
	main()