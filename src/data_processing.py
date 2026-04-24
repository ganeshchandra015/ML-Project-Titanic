import numpy as np
import pandas as pd


TITLE_MAPPING = {
	"Mr": "Mr",
	"Mrs": "Mrs",
	"Miss": "Miss",
	"Master": "Master",
	"Don": "Royalty",
	"Rev": "Officer",
	"Dr": "Officer",
	"Mme": "Mrs",
	"Ms": "Mrs",
	"Major": "Officer",
	"Lady": "Royalty",
	"Sir": "Royalty",
	"Mlle": "Miss",
	"Col": "Officer",
	"Capt": "Officer",
	"Countess": "Royalty",
	"Jonkheer": "Royalty",
	"Dona": "Royalty",
}

AGE_BINS = [0, 12, 18, 35, 60, 100]
AGE_LABELS = ["Child", "Teen", "Young_Adult", "Adult", "Senior"]
FARE_BIN_LABELS = ["Low", "Medium", "High", "Very_High"]
VALID_WORKFLOWS = {"baseline", "experiment_1", "experiment_2"}


def build_preprocessing_rules(features: pd.DataFrame, workflow: str = "baseline") -> dict:
	if workflow not in VALID_WORKFLOWS:
		raise ValueError(f"Unsupported workflow: {workflow}")

	rules = {
		"workflow": workflow,
		"median_age": features["Age"].median(),
		"mode_embarked": features["Embarked"].mode()[0],
		"median_fare": features["Fare"].median(),
		"title_mapping": TITLE_MAPPING,
		"age_bins": AGE_BINS,
		"age_labels": AGE_LABELS,
		"fare_bin_labels": FARE_BIN_LABELS,
	}

	if workflow in {"experiment_1", "experiment_2"}:
		_, fare_bin_edges = pd.qcut(
			features["Fare"].fillna(rules["median_fare"]),
			q=4,
			retbins=True,
			duplicates="drop",
		)
		fare_bin_edges[0] = -np.inf
		fare_bin_edges[-1] = np.inf
		rules["fare_bin_edges"] = fare_bin_edges

	return rules


def transform_features(features: pd.DataFrame, rules: dict) -> pd.DataFrame:
	workflow = rules["workflow"]
	transformed = features.copy()

	transformed["Age"] = transformed["Age"].fillna(rules["median_age"])
	transformed["Embarked"] = transformed["Embarked"].fillna(rules["mode_embarked"])
	transformed["Fare"] = transformed["Fare"].fillna(rules["median_fare"])

	transformed["Family_Size"] = transformed["SibSp"] + transformed["Parch"] + 1
	transformed["Is_Alone"] = (transformed["Family_Size"] == 1).astype(int)
	transformed["Title"] = transformed["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
	transformed["Title"] = transformed["Title"].replace(rules["title_mapping"])
	transformed["Title"] = transformed["Title"].fillna("Other")

	if workflow in {"experiment_1", "experiment_2"}:
		transformed["Age_Bin"] = pd.cut(
			transformed["Age"],
			bins=rules["age_bins"],
			labels=rules["age_labels"],
		)
		transformed["Fare_Bin"] = pd.cut(
			transformed["Fare"],
			bins=rules["fare_bin_edges"],
			labels=rules["fare_bin_labels"],
			include_lowest=True,
		)

	if workflow == "experiment_2":
		transformed["Has_Cabin"] = transformed["Cabin"].notnull().astype(int)

	columns_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
	if workflow in {"experiment_1", "experiment_2"}:
		columns_to_drop.extend(["Age", "Fare"])

	transformed.drop(columns=columns_to_drop, inplace=True)
	return transformed


def one_hot_encode(features: pd.DataFrame) -> pd.DataFrame:
	return pd.get_dummies(features, drop_first=True)


def align_to_reference(features: pd.DataFrame, reference_columns: pd.Index) -> pd.DataFrame:
	return features.reindex(columns=reference_columns, fill_value=0)
