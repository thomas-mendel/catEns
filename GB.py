# sklearn
from sklearn.ensemble import GradientBoostingClassifier

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


def instantiate_GradientBoostingClassifier_tuned(
    trial: Trial,
) -> GradientBoostingClassifier:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 800),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1, log=True),
        "subsample": trial.suggest_float("subsample", 0.1, 1),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_split": trial.suggest_float("min_samples_split", 0.01, 1),
        "min_samples_leaf": trial.suggest_float("min_sampls_leaf", 0.01, 0.5),
        "min_weight_fraction_leaf": trial.suggest_float(
            "min_weight_fraction_leaf", 1e-20, 0.5, log=True
        ),
        "min_impurity_decrease": trial.suggest_float(
            "min_impurity_decrease", 1e-20, 0.3, log=True
        ),
        "max_features": trial.suggest_float("max_features", 0.01, 1, log=True),
        "random_state": SEED,
    }
    return GradientBoostingClassifier(**params)


with MyTimer():
    experiment_name = "GB_tuned"
    mlflow.set_experiment(experiment_name)
    run_id = None
    run_name = "GB_tuned"
    tags = "GB_tuned"

    crossvalidate_pipeline(
        instantiate_GradientBoostingClassifier_tuned,
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
