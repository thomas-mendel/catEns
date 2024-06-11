# sklearn
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric

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


def instantiate_TabNet_tuned(trial: Trial) -> TabNetClassifier:
    n_a_d = trial.suggest_int("n_a_d", 20, 76)
    weight_decay = trial.suggest_float("weight_decay", 0.3, 0.6)
    # max_epochs: 30-60

    params = {
        "n_a": n_a_d,
        "n_d": n_a_d,
        "optimizer_params": {"lr": 2e-2, "weight_decay": weight_decay},
        "n_steps": trial.suggest_int("n_steps", 1, 8, step=1),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 5e-10, 0.01, log=True),
        "momentum": trial.suggest_float("momentum", 0.6, 0.98),
        "mask_type": trial.suggest_categorical("mask_type", ["entmax", "sparsemax"]),
        "n_shared": trial.suggest_int("n_shared", 1, 6),
        "n_independent": trial.suggest_int("n_independent", 1, 6),
        "seed": SEED,
        "verbose": 0,
    }
    return TabNetClassifier(**params)


experiment_name = "TN_tuned"
mlflow.set_experiment(experiment_name)
run_id = None
run_name = "TN_tuned"
tags = "TN_tuned"

crossvalidate_pipeline_TabNet(
    instantiate_TabNet_tuned,
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
