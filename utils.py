# sklearn
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    cohen_kappa_score,
    matthews_corrcoef,
    brier_score_loss,
)


# statistics
from numpy import mean
from numpy import std


# others
import numpy as np
import pandas as pd
import mlflow
import joblib
import time
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
import shap


class MyTimer:
    def __init__(self):
        self.start = time.perf_counter()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.perf_counter()
        runtime = end - self.start
        print(f"The steps took {runtime} seconds to complete.")


# decimal normalization
def scale_decimal(df: pd.DataFrame) -> pd.DataFrame:
    max_abs = np.max(np.abs(df))
    scaling_factor = np.ceil(np.log10(max_abs))
    scaled_data = df / 10**scaling_factor
    return scaled_data


# tanh normalization
def scale_tanh(df):
    df_ZSN = StandardScaler().fit_transform(df)
    scaled_data = 0.5 * (np.tanh(0.01 * df_ZSN) + 1)
    return scaled_data


# sigmoid normalization
def scale_sigmoid(df):
    df_ZSN = StandardScaler().fit_transform(df)
    scaled_data = 1 / (1 + np.exp(-df_ZSN))
    return scaled_data


DecimalScaler = FunctionTransformer(scale_decimal)
TanhScaler = FunctionTransformer(scale_tanh)
SigmoidScaler = FunctionTransformer(scale_sigmoid)


# OptionalPowerTranforer
class OptionalPowerTransformer(PowerTransformer):
    def __init__(self, method, standardize, use_transformer=True):
        super().__init__(method, standardize=standardize)
        self.use_transformer = use_transformer

    def fit(self, X, y=None):
        if self.use_transformer:
            return PowerTransformer.fit(self, X, y)
        else:
            return self

    def transform(self, X, y=None):
        if self.use_transformer:
            return PowerTransformer.transform(self, X)
        else:
            return np.array(X)

    def fit_transform(self, X, y=None):
        if self.use_transformer:
            return PowerTransformer.fit_transform(self, X, y)
        else:
            return np.array(X)


# instantiations
def instantiate_transformer(trial: Trial):
    params = {
        "use_transformer": trial.suggest_categorical("use_transformer", [True, False]),
        "method": "yeo-johnson",
        "standardize": False,
    }
    return OptionalPowerTransformer(**params)


def instantiate_normalizer(trial: Trial):
    normalizer = trial.suggest_categorical(
        "normalizer",
        [
            "StandardScaler",
            "MinMaxScaler",
            "DecimalScaler",
            "TanhScaler",
            "SigmoidScaler",
        ],
    )
    if normalizer == "StandardScaler":
        normalizer = StandardScaler()
    if normalizer == "MinMaxScaler":
        normalizer = MinMaxScaler()
    if normalizer == "DecimalScaler":
        normalizer = DecimalScaler
    if normalizer == "TanhScaler":
        normalizer = TanhScaler
    if normalizer == "SigmoidScaler":
        normalizer = SigmoidScaler
    return normalizer


class Slicer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold, feature_rank):
        self.feature_rank = feature_rank
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[X.columns[self.feature_rank <= self.threshold]]


def instantiate_slicer(trial: Trial):
    params = {
        "threshold": trial.suggest_int("threshold", 1, 200, log=True),
        "feature_rank": feature_rank,
    }
    return Slicer(**params)


def instantiate_pipeline(trial: Trial, instantiate_algorithm):

    pipeline = Pipeline(
        steps=[
            ("slicer", instantiate_slicer(trial)),
            ("transformer", instantiate_transformer(trial)),
            ("normalizer", instantiate_normalizer(trial)),
            ("algorithm", instantiate_algorithm(trial)),
        ]
    )
    return pipeline


# objective function
def objective(
    trial: Trial,
    instantiate_algorithm,
    X: pd.DataFrame,
    y: pd.Series,
    cv_inner,
    cv_inner_selection,
    feature_ranks,
    outer_fold_number=None,
) -> float:

    train_scores = []
    val_scores = []
    # one inner stritified cross validation loop
    for j, idx in enumerate(cv_inner):
        # slice data into fold
        train_idx, val_idx = idx
        assert (
            val_idx == cv_inner_selection[outer_fold_number][j][1]
        ).all(), "The validation indexes are not matched"
        X_train = X.iloc[train_idx, :]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx, :]
        y_val = y.iloc[val_idx]

        # instantiate pipeline
        global feature_rank
        feature_rank = feature_ranks[outer_fold_number][j]
        pipeline = instantiate_pipeline(trial, instantiate_algorithm)

        # fit and predict the pipeline
        pipeline.fit(X_train, y_train)
        train_pred = pipeline.predict_proba(X_train)[:, 1]
        val_pred = pipeline.predict_proba(X_val)[:, 1]

        # append the AUC scores of train and val datasets
        train_scores.append(roc_auc_score(y_train, train_pred))
        val_scores.append(roc_auc_score(y_val, val_pred))

    # write the mean and std of the AUC score into the trial
    trial.set_user_attr("train_mean", mean(train_scores))
    trial.set_user_attr("train_std", std(train_scores))
    trial.set_user_attr("val_mean", mean(val_scores))
    trial.set_user_attr("val_std", std(val_scores))

    # log meta data
    trial.set_user_attr("outer_fold_number", outer_fold_number)

    return np.mean(val_scores)


# Crossvalidate pipeline
def crossvalidate_pipeline(
    instantiate_algorithm,
    X: pd.DataFrame,
    y: pd.Series,
    cv_outer,
    run_name: str,
    run_id: str,
    tags: str,
    n_trials: int,
    cv_inner_selection=None,
    feature_ranks=None,
    experiment_name: str = "pilot",
):

    with mlflow.start_run(run_name=run_name, run_id=run_id) as run:
        mlflow.set_tags({"Model": tags})

        dev_scores = []
        test_scores = []

        for i, outer in enumerate(cv_outer):
            print(f"\n==================== outer fold {i} ====================")
            # Slice the data using cv_outer indices
            dev_idx, test_idx = outer
            X_dev = X.iloc[dev_idx, :].copy()
            y_dev = y.iloc[dev_idx].copy()
            X_test = X.iloc[test_idx, :].copy()
            y_test = y.iloc[test_idx].copy()

            run_name_nested = f"{run_name}_ofold_{i}"
            with mlflow.start_run(run_name=run_name_nested, nested=True) as run_nested:
                # inner fold indexing
                cv_inner = []
                skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                for train_idx, val_idx in skf_inner.split(X_dev, y_dev):
                    cv_inner.append((train_idx, val_idx))

                # hyperparameter tuning inner fold
                study_name = f"optimize_study_{run_name_nested}"
                study = optuna.create_study(
                    sampler=TPESampler(seed=42 + i),
                    direction="maximize",
                    study_name=study_name,
                    load_if_exists=False,
                )
                study.optimize(
                    lambda trial: objective(
                        trial=trial,  # Trial object is created here.
                        instantiate_algorithm=instantiate_algorithm,
                        X=X_dev,
                        y=y_dev,
                        cv_inner=cv_inner,
                        cv_inner_selection=cv_inner_selection,
                        feature_ranks=feature_ranks,
                        outer_fold_number=i,
                    ),
                    n_trials=n_trials,
                )

                # save study (every outer fold has a study about the dev)
                file_name = rf"{run_name_nested}_study.pkl"
                joblib.dump(study, file_name)
                mlflow.log_artifact(file_name)

                # retrieve and refit model on whole dev set
                best_trial = study.best_trial
                pipeline_best = instantiate_pipeline(best_trial, instantiate_algorithm)
                pipeline_best.fit(X_dev, y_dev)

                # evaluate the best model
                dev_pred = pipeline_best.predict_proba(X_dev)[:, 1]
                dev_score = roc_auc_score(y_dev, dev_pred)
                test_pred = pipeline_best.predict_proba(X_test)[:, 1]
                test_score = roc_auc_score(y_test, test_pred)
                dev_scores.append(dev_score)
                test_scores.append(test_score)

                # save the best pipeline
                file_name = rf"{run_name_nested}_best.pkl"
                joblib.dump(pipeline_best, filename=file_name, compress="gzip")
                mlflow.log_artifact(file_name)

                # set tags
                mlflow.set_tags({"Model": tags})

                # log best params
                mlflow.log_params(study.best_params)

                # log test metrics
                mlflow.log_metrics({"dev_score": dev_score, "test_score": test_score})

                # log and save each trial into data frame
                file_name = rf"{run_name_nested}_trials_dataframe.csv"
                trials_dataframe = study.trials_dataframe()
                trials_dataframe.columns = trials_dataframe.columns.str.replace(
                    "test", "val"
                )
                trials_dataframe["outer_fold_number"] = i
                trials_dataframe["algorithm"] = tags
                trials_dataframe.to_csv(file_name, sep=";")
                mlflow.log_artifact(file_name)

                # save optimization history
                figure = optuna.visualization.plot_optimization_history(study)
                file_name = rf"{run_name_nested}_optimization_history.html"
                figure.write_html(file_name)
                mlflow.log_artifact(file_name)

                # save param importance
                if run_name != "Dummy" and run_name != "Dummy_tuned":
                    figure = optuna.visualization.plot_param_importances(study)
                    file_name = rf"{run_name_nested}_hyperparameter_importance.html"
                    figure.write_html(file_name)
                    mlflow.log_artifact(file_name)

            # log outer fold summary metrics
            mlflow.log_metrics(
                {
                    "dev_mean": mean(dev_scores),
                    "dev_std": std(dev_scores),
                    "test_mean": mean(test_scores),
                    "test_std": std(test_scores),
                }
            )


# objective function
def objective_TabNet(
    trial: Trial,
    instantiate_algorithm,
    X: pd.DataFrame,
    y: pd.Series,
    cv_inner,
    cv_inner_selection,
    feature_ranks,
    outer_fold_number=None,
) -> float:

    train_scores = []
    val_scores = []
    # one inner stritified cross validation loop
    for j, idx in enumerate(cv_inner):
        # slice data into fold
        train_idx, val_idx = idx
        assert (
            val_idx == cv_inner_selection[outer_fold_number][j][1]
        ).all(), "The validation indexes are not matched"
        X_train = X.iloc[train_idx, :]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx, :]
        y_val = y.iloc[val_idx]

        # instantiate pipeline
        global feature_rank
        feature_rank = feature_ranks[outer_fold_number][j]
        pipeline = instantiate_pipeline(trial, instantiate_algorithm)

        fit_args = {
            pipeline.steps[-1][0]
            + "__virtual_batch_size": trial.suggest_int(
                "virtual_batch_size", 8, 100, step=4
            ),
            pipeline.steps[-1][0]
            + "__batch_size": trial.suggest_int("batch_size", 100, 152, step=4),
            pipeline.steps[-1][0]
            + "__max_epochs": trial.suggest_int("max_epochs", 30, 60),
            pipeline.steps[-1][0] + "__patience": trial.suggest_int("patience", 7, 10),
            pipeline.steps[-1][0] + "__eval_metric": "auc",
        }

        # fit and predict the pipeline
        pipeline.fit(X_train, y_train, **fit_args)

        train_pred = pipeline.predict_proba(X_train)[:, 1]
        val_pred = pipeline.predict_proba(X_val)[:, 1]

        # append the AUC scores of train and val datasets
        train_scores.append(roc_auc_score(y_train, train_pred))
        val_scores.append(roc_auc_score(y_val, val_pred))

    # write the mean and std of the AUC score into the trial
    trial.set_user_attr("train_mean", mean(train_scores))
    trial.set_user_attr("train_std", std(train_scores))
    trial.set_user_attr("val_mean", mean(val_scores))
    trial.set_user_attr("val_std", std(val_scores))

    # log meta data
    trial.set_user_attr("outer_fold_number", outer_fold_number)

    return np.mean(val_scores)


# Crossvalidate pipeline
def crossvalidate_pipeline_TabNet(
    instantiate_algorithm,
    X: pd.DataFrame,
    y: pd.Series,
    cv_outer,
    run_name: str,
    run_id: str,
    tags: str,
    n_trials: int,
    cv_inner_selection=None,
    feature_ranks=None,
    experiment_name: str = "pilot",
):

    with mlflow.start_run(run_name=run_name, run_id=run_id) as run:
        mlflow.set_tags({"Model": tags})

        dev_scores = []
        test_scores = []

        for i, outer in enumerate(cv_outer):
            print(f"\n==================== outer fold {i} ====================")
            # Slice the data using cv_outer indices
            dev_idx, test_idx = outer
            X_dev = X.iloc[dev_idx, :].copy()
            y_dev = y.iloc[dev_idx].copy()
            X_test = X.iloc[test_idx, :].copy()
            y_test = y.iloc[test_idx].copy()

            run_name_nested = f"{run_name}_ofold_{i}"
            with mlflow.start_run(run_name=run_name_nested, nested=True) as run_nested:
                # inner fold indexing
                cv_inner = []
                skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                for train_idx, val_idx in skf_inner.split(X_dev, y_dev):
                    cv_inner.append((train_idx, val_idx))

                # hyperparameter tuning inner fold
                study_name = f"optimize_study_{run_name_nested}"
                study = optuna.create_study(
                    sampler=TPESampler(seed=42 + i),
                    direction="maximize",
                    study_name=study_name,
                    load_if_exists=False,
                )
                study.optimize(
                    lambda trial: objective_TabNet(
                        trial=trial,  # Trial object is created here.
                        instantiate_algorithm=instantiate_algorithm,
                        X=X_dev,
                        y=y_dev,
                        cv_inner=cv_inner,
                        cv_inner_selection=cv_inner_selection,
                        feature_ranks=feature_ranks,
                        outer_fold_number=i,
                    ),
                    n_trials=n_trials,
                )

                # save study (every outer fold has a study about the dev)
                file_name = rf"{run_name_nested}_study.pkl"
                joblib.dump(study, file_name)
                mlflow.log_artifact(file_name)

                # retrieve and refit model on whole dev set
                best_trial = study.best_trial
                pipeline_best = instantiate_pipeline(best_trial, instantiate_algorithm)
                fit_args = {
                    pipeline_best.steps[-1][0]
                    + "__virtual_batch_size": best_trial.params["virtual_batch_size"],
                    pipeline_best.steps[-1][0]
                    + "__batch_size": best_trial.params["batch_size"],
                    pipeline_best.steps[-1][0]
                    + "__max_epochs": best_trial.params["max_epochs"],
                    pipeline_best.steps[-1][0] + "__eval_metric": "auc",
                    pipeline_best.steps[-1][0]
                    + "__patience": best_trial.params["patience"],
                }
                pipeline_best.fit(X_dev, y_dev, **fit_args)

                # evaluate the best model
                dev_pred = pipeline_best.predict_proba(X_dev)[:, 1]
                dev_score = roc_auc_score(y_dev, dev_pred)
                test_pred = pipeline_best.predict_proba(X_test)[:, 1]
                test_score = roc_auc_score(y_test, test_pred)
                dev_scores.append(dev_score)
                test_scores.append(test_score)

                # save the best pipeline
                file_name = rf"{run_name_nested}_best.pkl"
                joblib.dump(pipeline_best, filename=file_name, compress="gzip")
                mlflow.log_artifact(file_name)  # pipeline is fitted

                # set tags
                mlflow.set_tags({"Model": tags})

                # log best params
                mlflow.log_params(study.best_params)

                # log test metrics
                mlflow.log_metrics({"dev_score": dev_score, "test_score": test_score})

                # log and save each trial into data frame
                file_name = rf"{run_name_nested}_trials_dataframe.csv"
                trials_dataframe = study.trials_dataframe()
                trials_dataframe.columns = trials_dataframe.columns.str.replace(
                    "test", "val"
                )
                trials_dataframe["outer_fold_number"] = i
                trials_dataframe["algorithm"] = tags
                trials_dataframe.to_csv(file_name, sep=";")
                mlflow.log_artifact(file_name)

                # save optimization history
                figure = optuna.visualization.plot_optimization_history(study)
                file_name = rf"{run_name_nested}_optimization_history.html"
                figure.write_html(file_name)
                mlflow.log_artifact(file_name)

                # save param importance
                if run_name != "Dummy" and run_name != "Dummy_tuned":
                    figure = optuna.visualization.plot_param_importances(study)
                    file_name = rf"{run_name_nested}_hyperparameter_importance.html"
                    figure.write_html(file_name)
                    mlflow.log_artifact(file_name)

            # log outer fold summary metrics
            mlflow.log_metrics(
                {
                    "dev_mean": mean(dev_scores),
                    "dev_std": std(dev_scores),
                    "test_mean": mean(test_scores),
                    "test_std": std(test_scores),
                }
            )


def shap_feature_ranking(data, shap_values, columns=[]):
    """from https://stackoverflow.com/questions/65534163/get-a-feature-importance-from-shap-values"""
    if not columns:
        columns = data.columns.tolist()
    c_idxs = []
    for column in columns:
        c_idxs.append(data.columns.get_loc(column))
    if isinstance(shap_values, list):
        means = [
            np.abs(shap_values[class_][:, c_idxs]).mean(axis=0)
            for class_ in range(len(shap_values))
        ]
        shap_means = np.sum(np.column_stack(means), 1)
    else:
        assert (
            len(shap_values.shape) == 2
        ), "Expected two-dimensional shap values array."
        shap_means = np.abs(shap_values).mean(axis=0)
    df_ranking = (
        pd.DataFrame({"feature": columns, "mean_shap_value": shap_means})
        .sort_values(by="mean_shap_value", ascending=False)
        .reset_index(drop=True)
    )
    return df_ranking


class Evaluation:

    def __init__(
        self, pipelines, X_tests, y_tests, tag="SVC", cv=None, X_devs=None, y_devs=None
    ):
        self.pipelines = pipelines
        self.X_devs = X_devs
        self.y_devs = y_devs
        self.X_tests = X_tests
        self.y_tests = y_tests
        self.tag = tag
        self.cv = cv

    def compute_metrics(self):
        result_list = []
        if isinstance(self.pipelines, list):
            for i, pipeline in enumerate(self.pipelines):

                y_test_pred = pipeline.predict(self.X_tests[i])
                y_test_proba = pipeline.predict_proba(self.X_tests[i])
                y_dev_pred = pipeline.predict(self.X_devs[i])
                y_dev_proba = pipeline.predict_proba(self.X_devs[i])

                test_accuracy = accuracy_score(self.y_tests[i], y_test_pred)
                test_f1 = f1_score(self.y_tests[i], y_test_pred)
                test_auc_roc = roc_auc_score(self.y_tests[i], y_test_proba[:, 1])
                test_Recall = recall_score(self.y_tests[i], y_test_pred)
                test_Precision = precision_score(self.y_tests[i], y_test_pred)
                test_Kappa = cohen_kappa_score(self.y_tests[i], y_test_pred)
                test_MCC = matthews_corrcoef(self.y_tests[i], y_test_pred)
                test_BS = brier_score_loss(self.y_tests[i], y_test_proba[:, 1])

                dev_accuracy = accuracy_score(self.y_devs[i], y_dev_pred)
                dev_f1 = f1_score(self.y_devs[i], y_dev_pred)
                dev_auc_roc = roc_auc_score(self.y_devs[i], y_dev_proba[:, 1])
                dev_Recall = recall_score(self.y_devs[i], y_dev_pred)
                dev_Precision = precision_score(self.y_devs[i], y_dev_pred)
                dev_Kappa = cohen_kappa_score(self.y_devs[i], y_dev_pred)
                dev_MCC = matthews_corrcoef(self.y_devs[i], y_dev_pred)
                dev_BS = brier_score_loss(self.y_devs[i], y_dev_proba[:, 1])

                if self.tag != "TN":
                    if isinstance(pipeline, sklearn.pipeline.Pipeline):
                        if self.tag != "LGBM_nine" and self.tag != "LGBM_two":
                            bool_idx = pipeline[0].feature_rank <= pipeline[0].threshold
                            num_feats = sum(bool_idx)
                        else:
                            num_feats = self.X_devs[i].shape[1]
                    else:
                        num_feats = self.X_devs[i].shape[1]
                else:
                    num_feats = self.X_devs[i].shape[1]

                result_list.append(
                    (
                        self.tag,
                        i,
                        num_feats,
                        dev_accuracy,
                        dev_f1,
                        dev_auc_roc,
                        dev_Recall,
                        dev_Precision,
                        dev_Kappa,
                        dev_MCC,
                        dev_BS,
                        test_accuracy,
                        test_f1,
                        test_auc_roc,
                        test_Recall,
                        test_Precision,
                        test_Kappa,
                        test_MCC,
                        test_BS,
                    )
                )
            result_df = pd.DataFrame(
                result_list,
                columns=(
                    "algorithm",
                    "fold",
                    "number of features",
                    "dev_accuracy",
                    "dev_f1",
                    "dev_auc_roc",
                    "dev_Recall",
                    "dev_Precision",
                    "dev_Kappa",
                    "dev_MCC",
                    "dev_BS",
                    "test_accuracy",
                    "test_f1",
                    "test_ROC",
                    "test_Recall",
                    "test_Precision",
                    "test_Kappa",
                    "test_MCC",
                    "test_BS",
                ),
            )
        else:
            pass
        return result_df

    def get_shap_ranks(self, algorithm_fold):
        fold_number = algorithm_fold.loc[self.tag].values[0]
        best_X_dev = self.X_devs[fold_number]
        best_pipeline = self.pipelines[fold_number]
        bool_idx = best_pipeline[0].feature_rank <= best_pipeline[0].threshold
        best_genes = best_X_dev.loc[:, bool_idx].columns
        best_X_dev = pd.DataFrame(best_pipeline[0:3].transform(best_X_dev))
        best_X_dev.columns = best_genes
        explainer = shap.KernelExplainer(
            best_pipeline[-1].predict_proba, best_X_dev
        )  # SEED can be fixed here.
        shap_values = explainer.shap_values(best_X_dev)
        shap_ranking = shap_feature_ranking(data=best_X_dev, shap_values=shap_values)
        joblib.dump(explainer, self.tag + "_explainer.pkl.gz", compress=8)
        joblib.dump(shap_values, self.tag + "_shapvalues.pkl.gz", compress=8)
        joblib.dump(best_X_dev, self.tag + "_bestXdev.pkl.gz", compress=8)
        shap_ranking.to_csv(self.tag + "_generank.csv", index=False)
        return shap_ranking
