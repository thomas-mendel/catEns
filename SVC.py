# sklearn
from sklearn.svm import SVC

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


def instantiate_SVC_tuned(trial: Trial) -> SVC:
    params = {
        "C": trial.suggest_float("C", 1e-10, 500, log=True),
        "kernel": trial.suggest_categorical(
            "kernel", ["linear", "rbf", "sigmoid", "poly"]
        ),
        "degree": trial.suggest_int("degree", 1, 5),
        "gamma": trial.suggest_float("gamma", 1e-5, 1, log=True),
        "coef0": trial.suggest_float("coef0", 0.001, 100, log=True),
        "shrinking": trial.suggest_categorical("shringking", [True, False]),
        "tol": trial.suggest_float("tol", 1e-6, 6, log=True),
        "random_state": SEED,
        "probability": True,
        "cache_size": 7000,
        "max_iter": 1000,
    }
    return SVC(**params)


with MyTimer():
    experiment_name = "SVC_tuned"
    mlflow.set_experiment(experiment_name)
    run_id = None
    run_name = "SVC_tuned"
    tags = "SVC_tuned"

    crossvalidate_pipeline(
        instantiate_SVC_tuned,
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
