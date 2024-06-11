# sklearn
from sklearn.naive_bayes import GaussianNB

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


def instantiate_GaussianNB_tuned(trial: Trial) -> GaussianNB:
    params = {
        "var_smoothing": trial.suggest_float("var_smoothing", 0, 1.0),
        "priors": None,
    }
    return GaussianNB(**params)


with MyTimer():
    experiment_name = "GNB_tuned"
    mlflow.set_experiment(experiment_name)
    run_id = None
    run_name = "GNB_tuned"
    tags = "GNB_tuned"

    crossvalidate_pipeline(
        instantiate_GaussianNB_tuned,
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
