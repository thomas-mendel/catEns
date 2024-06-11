# sklearn
from sklearn.ensemble import AdaBoostClassifier

# others
import mlflow
from optuna import Trial

# self-defined
from utils import *


SEED = 42

with open(file="cv_outer.pickle", mode="rb") as file:
    cv_outer = joblib.load(file)
with open(file="cv_inner_selection.pickle", mode="rb") as file:
    cv_inner_selection = joblib.load(file)
cv_inner_selection
with open(file="feature_ranks.pickle", mode="rb") as file:
    feature_ranks = joblib.load(file)


df = pd.read_csv("df.csv", index_col=0)
X = df.drop(["target", "dataset"], axis=1).copy()
y = df["target"].replace({"Bystander": 1, "Control": 0})


def instantiate_AdaBoostClassifier_tuned(trial: Trial) -> AdaBoostClassifier:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 10, log=True),
        "algorithm": trial.suggest_categorical("algorithm", ["SAMME", "SAMME.R"]),
        "random_state": SEED,
    }
    return AdaBoostClassifier(**params)


experiment_name = "AB_tuned"
mlflow.set_experiment(experiment_name)
run_id = None
run_name = "AB_tuned"
tags = "AB_tuned"

crossvalidate_pipeline(
    instantiate_AdaBoostClassifier_tuned,
    X,
    y,
    cv_outer,
    run_name=run_name,
    run_id=run_id,
    tags=tags,
    n_trials=200,
    cv_inner_selection=cv_inner_selection,
    feature_ranks=feature_ranks,
    experiment_name=experiment_name,
)
