# sklearn
from catboost import CatBoostClassifier

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


def instantiate_CatBoostClassifier_tuned(trial: Trial) -> CatBoostClassifier:
    params = {
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1, log=True),
        "iterations": trial.suggest_int("iterations", 50, 500),
        "depth": trial.suggest_int("max_depth", 4, 10),
        "random_strength": trial.suggest_float("random_strength", 0.01, 20),
        "border_count": 254,
        "min_child_samples": trial.suggest_int("min_child_samples", 6, 50),
        "colsample_bylevel": trial.suggest_float(
            "colsample_bylevel", 0.05, 1, log=True
        ),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bernoulli", "MVS"]
        ),
        "boosting_type": "Plain",
        "eval_metric": "AUC",
        "random_state": SEED,
        "verbose": False,
        "subsample": trial.suggest_float("subsample", 0.1, 1),
        "grow_policy": "SymmetricTree",
        "thread_count": -1,
    }
    return CatBoostClassifier(**params)


with MyTimer():
    experiment_name = "CB_tuned"
    mlflow.set_experiment(experiment_name)
    run_id = None
    run_name = "CB_tuned"
    tags = "CB_tuned"

    crossvalidate_pipeline(
        instantiate_CatBoostClassifier_tuned,
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
