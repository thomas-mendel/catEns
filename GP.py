# sklearn
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    Matern,
    WhiteKernel,
    ConstantKernel,
    RationalQuadratic,
    RBF,
)


# others
import mlflow
from optuna import Trial
import warnings

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


def instantiate_GaussianProcessClassifier_tuned(
    trial: Trial,
) -> GaussianProcessClassifier:
    params = {
        "kernel": trial.suggest_categorical(
            "kernel",
            [
                1 * RBF(),
                1 * Matern(),
                1 * RationalQuadratic(),
                1 * WhiteKernel(),
                1 * ConstantKernel(),
            ],
        ),  # 1*dotproduct() was removed because of possible errors.
        "n_restarts_optimizer": trial.suggest_int("n_restarts_optimizer", 0, 50),
        "max_iter_predict": trial.suggest_int("max_iter_predict", 20, 200),
        "n_jobs": -1,
        "random_state": SEED,
    }
    return GaussianProcessClassifier(**params)


warnings.simplefilter("ignore")

with MyTimer():
    experiment_name = "GP_tuned"
    mlflow.set_experiment(experiment_name)
    run_id = None
    run_name = "GP_tuned"
    tags = "GP_tuned"

    crossvalidate_pipeline(
        instantiate_GaussianProcessClassifier_tuned,
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
