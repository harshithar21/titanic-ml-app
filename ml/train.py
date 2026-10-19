"""Train, evaluate and save the Titanic survival model.

Run from the project root:
    python -m ml.train             # train + save model + figures
    python -m ml.train --no-plots  # same, without matplotlib (used in Docker / Render builds)
"""
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ml.features import RAW_FEATURES, add_features

ROOT = Path(__file__).resolve().parent.parent
RS = 42

NUMERIC = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
CATEGORICAL = ["Sex", "Embarked"]
PASSTHROUGH = ["Pclass", "HasCabin", "IsAlone"]


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    return df[RAW_FEATURES], df["Survived"]


def build_pipeline(model) -> Pipeline:
    """Feature engineering + imputing + scaling + encoding + model in ONE object."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))])
    pre = ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL),
                             ("pass", "passthrough", PASSTHROUGH)])
    return Pipeline([("features", FunctionTransformer(add_features)), ("prep", pre), ("model", model)])


def candidates():
    cv = StratifiedKFold(5, shuffle=True, random_state=RS)
    return {
        "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=RS),
            GridSearchCV, {"model__C": [0.1, 1, 10]}),
        "Random Forest": (RandomForestClassifier(random_state=RS),
            RandomizedSearchCV, {"model__n_estimators": [100, 200, 300], "model__max_depth": [3, 4, 5, 6, 8, None],
                                 "model__min_samples_leaf": [1, 2, 4, 8], "model__max_features": ["sqrt", 0.5]}),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=RS),
            RandomizedSearchCV, {"model__n_estimators": [100, 200, 300], "model__learning_rate": [0.03, 0.05, 0.1],
                                 "model__max_depth": [2, 3, 4], "model__subsample": [0.7, 0.85, 1.0]}),
    }, cv


def run(data_path=None, models_dir=None, reports_dir=None, plots=True) -> dict:
    data_path = Path(data_path or ROOT / "data" / "titanic.csv")
    models_dir = Path(models_dir or ROOT / "models"); models_dir.mkdir(exist_ok=True, parents=True)
    reports_dir = Path(reports_dir or ROOT / "reports"); (reports_dir / "figures").mkdir(exist_ok=True, parents=True)

    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
    print(f"Data: {X.shape[0]} rows | train {len(X_train)} | test {len(X_test)} | survival rate {y.mean():.3f}")

    cands, cv = candidates()
    scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}
    baseline_rows, tuned_rows, tuned_models = [], [], {}
    for name, (est, SearchCls, grid) in cands.items():
        # 1) default settings, 5-fold CV on the training data only
        s = cross_validate(build_pipeline(est), X_train, y_train, cv=cv, scoring=scoring)
        baseline_rows.append({"model": name, **{k: float(s[f"test_{k}"].mean()) for k in scoring}})
        # 2) hyperparameter tuning, again using only the training data
        kw = {"n_iter": 15, "random_state": RS} if SearchCls is RandomizedSearchCV else {}
        search = SearchCls(build_pipeline(est), grid, cv=cv, scoring="roc_auc", n_jobs=-1, **kw).fit(X_train, y_train)
        tuned_models[name] = search.best_estimator_
        tuned_rows.append({"model": name, "cv_roc_auc": float(search.best_score_),
                           "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()}})
        print(f"{name:20s} default CV AUC {baseline_rows[-1]['roc_auc']:.4f} -> tuned {search.best_score_:.4f}")

    best_name = max(tuned_rows, key=lambda r: r["cv_roc_auc"])["model"]   # chosen by CV, not by the test set
    best = tuned_models[best_name]
    print("Selected model (best CV ROC-AUC):", best_name)

    # final, one-time evaluation on the untouched test set
    proba = best.predict_proba(X_test)[:, 1]; pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    test = {"accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred), "roc_auc": roc_auc_score(y_test, proba),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "majority_class_baseline_accuracy": float(1 - y_test.mean())}
    test = {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in test.items()}
    print("Test:", test)

    imp = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=RS)
    importance = dict(sorted(zip(X_test.columns, imp.importances_mean.round(4).tolist()), key=lambda kv: -kv[1]))

    # save the whole pipeline (preprocessing + model) as one file
    joblib.dump(best, models_dir / "titanic_pipeline.joblib")
    meta = {"model_name": best_name, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
            "features": RAW_FEATURES, "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "best_params": next(r["best_params"] for r in tuned_rows if r["model"] == best_name), "test_metrics": test}
    (models_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    metrics = {**meta, "cv_default": baseline_rows, "cv_tuned": tuned_rows, "permutation_importance": importance}
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print("Saved model ->", models_dir / "titanic_pipeline.joblib")

    if plots:
        make_plots(X, y, X_test, y_test, proba, pred, baseline_rows, tuned_rows, importance, reports_dir / "figures")
    return metrics


def make_plots(X, y, X_test, y_test, proba, pred, base, tuned, importance, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
    blue, orange = "#2E5597", "#E07B39"

    d = X.copy(); d["Survived"] = y
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
    d.groupby("Sex")["Survived"].mean().plot.bar(ax=ax[0, 0], color=blue, rot=0, title="Survival rate by sex")
    d.groupby("Pclass")["Survived"].mean().plot.bar(ax=ax[0, 1], color=blue, rot=0, title="Survival rate by class")
    d.groupby("Embarked")["Survived"].mean().plot.bar(ax=ax[1, 0], color=blue, rot=0, title="Survival rate by port")
    ax[1, 1].hist([d.loc[d.Survived == 0, "Age"].dropna(), d.loc[d.Survived == 1, "Age"].dropna()], bins=20,
                  stacked=True, color=[blue, orange], label=["Died", "Survived"]); ax[1, 1].legend(); ax[1, 1].set_title("Age distribution")
    plt.tight_layout(); plt.savefig(out / "eda_survival.png", dpi=130); plt.close()

    names = [r["model"] for r in base]; x = np.arange(len(names)); w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, [r["roc_auc"] for r in base], w, label="Default settings", color=blue)
    ax.bar(x + w / 2, [r["cv_roc_auc"] for r in tuned], w, label="Tuned", color=orange)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0.80, 0.90); ax.set_ylabel("5-fold CV ROC-AUC"); ax.legend(); ax.set_title("Model comparison (cross-validation)")
    plt.tight_layout(); plt.savefig(out / "model_comparison.png", dpi=130); plt.close()

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["Died", "Survived"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion matrix (test set)"); ax.grid(False); plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=130); plt.close()

    fpr, tpr, _ = roc_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.plot(fpr, tpr, color=blue, label=f"AUC = {roc_auc_score(y_test, proba):.3f}"); ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("ROC curve (test set)"); ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out / "roc_curve.png", dpi=130); plt.close()

    imp = pd.Series(importance).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    imp.plot.barh(ax=ax, color=blue); ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled"); ax.set_title("Permutation importance (test set)")
    plt.tight_layout(); plt.savefig(out / "permutation_importance.png", dpi=130); plt.close()
    print("Figures saved ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--no-plots", action="store_true")
    run(plots=not ap.parse_args().no_plots)
