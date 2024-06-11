# sklearn
from sklearn.linear_model import LogisticRegression

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


def instantiate_LogisticRegression_tuned(trial: Trial) -> LogisticRegression:
    params = {
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"]),
        "tol": trial.suggest_float("tol", 1e-10, 1, log=True),
        "C": trial.suggest_float("C", 1e-5, 50),
        "l1_ratio": trial.suggest_float("l1_ratio", 0, 1),
        "random_state": SEED,
        "solver": "saga",
        "max_iter": 1000,
        "n_jobs": -1,
    }
    return LogisticRegression(**params)


with MyTimer():
    experiment_name = "LR_tuned"
    mlflow.set_experiment(experiment_name)
    run_id = None
    run_name = "LR_tuned"
    tags = "LR_tuned"

    crossvalidate_pipeline(
        instantiate_LogisticRegression_tuned,
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
