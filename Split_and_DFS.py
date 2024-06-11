# sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# others
import pandas as pd
import numpy as np
from glob import glob
import time
import joblib
from boruta import BorutaPy

SEED = 42
outer_n_splits = 6
inner_n_splits = 5

df = pd.read_csv("df.csv", index_col=0)
X = df.drop(["target", "dataset"], axis=1).copy()
y = df["target"].replace({"Bystander": 1, "Control": 0})


# outer folds split
cv_outer = []
skf_outer = StratifiedKFold(n_splits=outer_n_splits, shuffle=True, random_state=SEED)
for dev_idx, test_idx in skf_outer.split(X, y):
    cv_outer.append((dev_idx, test_idx))
with open(file="cv_outer.pickle", mode="wb") as file:
    joblib.dump(cv_outer, file)

# feature selection
cv_inner_selection = []
for i, outer in enumerate(cv_outer):
    # Slice the data using cv_outer indices
    dev_idx, test_idx = outer
    X_dev = X.iloc[dev_idx, :].copy()
    y_dev = y.iloc[dev_idx].copy()
    X_test = X.iloc[test_idx, :].copy()
    y_test = y.iloc[test_idx].copy()

    cv_inner_ofold = []
    skf_inner = StratifiedKFold(
        n_splits=inner_n_splits, shuffle=True, random_state=SEED
    )
    for train_idx, val_idx in skf_inner.split(X_dev, y_dev):
        cv_inner_ofold.append((train_idx, val_idx))

    cv_inner_selection.append(cv_inner_ofold)

with open(file="cv_inner_selection.pickle", mode="wb") as file:
    joblib.dump(cv_inner_selection, file)


start_time = time.perf_counter()
# feature selection
feature_ranks = []

for i, outer_idx in enumerate(cv_inner_selection):
    print(f"\nouter fold {i} ======")

    feature_rank_inner = []
    for j, inner_idx in enumerate(outer_idx):
        print(f"\n=== inner fold {j} ===")
        train_idx, val_idx = inner_idx
        X_train = X.iloc[train_idx, :]
        y_train = y.iloc[train_idx]

        feature_selector = BorutaPy(
            verbose=0,
            estimator=RandomForestClassifier(random_state=SEED, n_jobs=-1, max_depth=5),
            n_estimators="auto",
            random_state=SEED,
            alpha=0.05,
            perc=100,
            max_iter=100,
        )
        feature_selector.fit(np.array(X_train), np.array(y_train))
        feature_rank_inner.append(feature_selector.ranking_)

    feature_ranks.append(feature_rank_inner)

end_time = time.perf_counter()
print(f"The time used for feature selection: {end_time-start_time}")

with open(file="feature_ranks.pickle", mode="wb") as file:
    joblib.dump(feature_ranks, file)
