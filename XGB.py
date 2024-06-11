# sklearn
from xgboost import XGBClassifier

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


def instantiate_XGBClassifier_tuned(trial: Trial) -> XGBClassifier:
    max_depth = trial.suggest_int("max_depth", 3, 12)

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 10, 800),
        "max_depth": max_depth,
        "subsample": trial.suggest_float("subsample", 0.1, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.1, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-5, 5, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 15),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-6, 30),
        "max_leaves": 2**max_depth,
        "eval_metric": "auc",
        "n_jobs": -1,
        "random_state": SEED,
        "verbosity": 0,
    }
    return XGBClassifier(**params)


with MyTimer():
    experiment_name = "XGB_tuned"
    mlflow.set_experiment(experiment_name)
    run_id = None
    run_name = "XGB_tuned"
    tags = "XGB_tuned"

    crossvalidate_pipeline(
        instantiate_XGBClassifier_tuned,
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
