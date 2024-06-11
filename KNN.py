# sklearn
from sklearn.neighbors import KNeighborsClassifier

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


def instantiate_KNeighborsClassifier_tuned(trial: Trial) -> KNeighborsClassifier:
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 100),
        "algorithm": trial.suggest_categorical(
            "algorithm", ["ball_tree", "kd_tree", "brute", "auto"]
        ),
        "metric": trial.suggest_categorical(
            "metric", ["euclidean", "manhattan", "minkowski", "chebyshev"]
        ),
        "leaf_size": trial.suggest_int("n_neighbors", 1, 100),
        "p": trial.suggest_int("p", 1, 10),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "n_jobs": -1,
    }
    return KNeighborsClassifier(**params)


with MyTimer():
    experiment_name = "KNN_tuned"
    mlflow.set_experiment(experiment_name)
    run_id = None
    run_name = "KNN_tuned"
    tags = "KNN_tuned"

    crossvalidate_pipeline(
        instantiate_KNeighborsClassifier_tuned,
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
