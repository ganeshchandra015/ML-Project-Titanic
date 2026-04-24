from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd


def train_logistic_regression(features, target, random_state: int = 42, max_iter: int = 1000):
	model = LogisticRegression(random_state=random_state, max_iter=max_iter)
	model.fit(features, target)
	return model


def predict_labels_and_probabilities(model, features):
	labels = model.predict(features)
	probabilities = model.predict_proba(features)[:, 1]
	return labels, probabilities


def evaluate_binary_classifier(target, predictions) -> dict:
	return {
		"accuracy": accuracy_score(target, predictions),
		"precision": precision_score(target, predictions),
		"recall": recall_score(target, predictions),
		"f1": f1_score(target, predictions),
	}


def extract_sorted_coefficients(model, feature_names) -> pd.DataFrame:
	coefficient_frame = pd.DataFrame(
		{
			"feature": feature_names,
			"coefficient": model.coef_[0],
		}
	)
	return coefficient_frame.sort_values(by="coefficient", ascending=False).reset_index(drop=True)

