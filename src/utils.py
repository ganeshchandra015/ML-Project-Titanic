from __future__ import annotations

import os
import pickle
from typing import Any

import pandas as pd


def ensure_directory(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def save_pickle(obj: Any, file_path: str) -> None:
	ensure_directory(os.path.dirname(file_path))
	with open(file_path, "wb") as file_handle:
		pickle.dump(obj, file_handle)


def load_pickle(file_path: str) -> Any:
	with open(file_path, "rb") as file_handle:
		return pickle.load(file_handle)


def build_submission_frame(passenger_ids: pd.Series, predictions) -> pd.DataFrame:
	return pd.DataFrame(
		{
			"PassengerId": passenger_ids,
			"Survived": predictions,
		}
	)


def save_submission(submission: pd.DataFrame, file_path: str) -> None:
	ensure_directory(os.path.dirname(file_path))
	submission.to_csv(file_path, index=False)
