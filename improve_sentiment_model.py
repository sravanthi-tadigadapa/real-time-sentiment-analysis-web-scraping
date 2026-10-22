import json
import os
import tempfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


LABELS = ["negative", "neutral", "positive"]
SCORE_TO_SENTIMENT = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
STORED_LABELS = {"0": "negative", "1": "neutral", "2": "positive"}
RANDOM_STATE = 42
CV_PER_CLASS = {"negative": 4_000, "neutral": 4_000, "positive": 8_000}


def read_amazon(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {"clean_text", "Score", "sentiment", "sentiment_label"}
    frame = pd.read_csv(
        path,
        usecols=lambda column: column in required,
        dtype={"clean_text": "string", "Score": "string", "sentiment": "string", "sentiment_label": "string"},
    )
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Amazon CSV is missing required columns: {', '.join(sorted(missing))}")

    frame["text"] = frame["clean_text"].fillna("").str.strip()
    frame["score"] = pd.to_numeric(frame["Score"], errors="coerce")
    frame["sentiment"] = frame["sentiment"].str.strip().str.lower()
    frame["sentiment_label"] = frame["sentiment_label"].str.strip()
    empty_count = int(frame["text"].eq("").sum())
    frame = frame.loc[frame["text"].ne("")].copy()
    frame["label"] = frame["score"].map(SCORE_TO_SENTIMENT)
    frame["stored_label"] = frame["sentiment_label"].map(STORED_LABELS)

    invalid_scores = frame["label"].isna()
    if invalid_scores.any():
        raise ValueError(f"Amazon CSV contains {int(invalid_scores.sum())} rows with invalid or missing scores.")
    score_conflicts = frame["label"].ne(frame["sentiment"])
    stored_conflicts = frame["label"].ne(frame["stored_label"])
    if score_conflicts.any() or stored_conflicts.any():
        raise ValueError(
            "Amazon score-derived labels conflict with sentiment or sentiment_label "
            f"({int(score_conflicts.sum())} and {int(stored_conflicts.sum())} rows)."
        )
    duplicate_rows = int(frame["text"].duplicated(keep=False).sum())
    conflicting_duplicate_texts = int(frame.groupby("text", sort=False)["label"].nunique().gt(1).sum())
    audit = {
        "rows_after_empty_text_exclusion": len(frame),
        "empty_text_rows": empty_count,
        "score_sentiment_conflicts": int(score_conflicts.sum()),
        "score_stored_label_conflicts": int(stored_conflicts.sum()),
        "duplicate_text_rows": duplicate_rows,
        "duplicate_texts_with_conflicting_labels": conflicting_duplicate_texts,
    }
    if conflicting_duplicate_texts:
        raise ValueError("Exact duplicate review text occurs with conflicting sentiment labels.")
    return frame[["text", "score", "sentiment", "sentiment_label", "label"]], audit


def read_clothing(path: Path) -> pd.DataFrame:
    required = {"Review Text", "Rating"}
    frame = pd.read_csv(
        path,
        usecols=lambda column: column in required,
        dtype={"Review Text": "string", "Rating": "string"},
    )
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Clothing CSV is missing required columns: {', '.join(sorted(missing))}")
    frame["text"] = frame["Review Text"].fillna("").str.strip()
    frame["rating"] = pd.to_numeric(frame["Rating"], errors="coerce")
    invalid_rating = frame["rating"].notna() & ~frame["rating"].isin(range(1, 6))
    if invalid_rating.any():
        raise ValueError(f"Clothing CSV contains {int(invalid_rating.sum())} ratings outside the 1-5 range.")
    frame = frame.loc[frame["text"].ne("") & frame["rating"].isin(range(1, 6))].copy()
    frame["label"] = frame["rating"].map(SCORE_TO_SENTIMENT)
    return frame[["text", "rating", "label"]]


def vectorizer_config(representation: str) -> dict:
    word_features = {
        "max_features": 70_000,
        "min_df": 2,
        "ngram_range": (1, 2),
        "strip_accents": "unicode",
        "sublinear_tf": True,
    }
    char_features = {
        "analyzer": "char",
        "max_features": 70_000,
        "min_df": 3,
        "ngram_range": (3, 5),
        "strip_accents": "unicode",
        "sublinear_tf": True,
    }
    if representation == "word_1_2":
        return {"type": "TfidfVectorizer", **word_features}
    if representation == "word_1_3":
        return {"type": "TfidfVectorizer", **{**word_features, "ngram_range": (1, 3)}}
    if representation == "char_3_5":
        return {"type": "TfidfVectorizer", **char_features}
    if representation == "word_char":
        return {
            "type": "FeatureUnion",
            "word": {"type": "TfidfVectorizer", **{**word_features, "max_features": 45_000}},
            "char": {"type": "TfidfVectorizer", **{**char_features, "max_features": 45_000}},
        }
    raise ValueError(f"Unknown representation: {representation}")


def make_transformer(representation: str):
    config = vectorizer_config(representation)
    if representation == "word_char":
        return FeatureUnion(
            [
                ("word", TfidfVectorizer(**{k: v for k, v in config["word"].items() if k != "type"})),
                ("char", TfidfVectorizer(**{k: v for k, v in config["char"].items() if k != "type"})),
            ],
            n_jobs=1,
        )
    return TfidfVectorizer(**{k: v for k, v in config.items() if k != "type"})


def make_classifier(classifier_name: str, parameter: float, baseline: bool = False):
    if classifier_name == "Logistic Regression":
        return LogisticRegression(
            C=parameter,
            class_weight=None if baseline else "balanced",
            max_iter=1500,
            random_state=RANDOM_STATE,
        )
    if classifier_name == "LinearSVC":
        return LinearSVC(
            C=parameter,
            class_weight=None if baseline else "balanced",
            max_iter=7000,
            random_state=RANDOM_STATE,
        )
    if classifier_name == "MultinomialNB":
        return MultinomialNB(alpha=parameter, fit_prior=False)
    raise ValueError(f"Unknown classifier: {classifier_name}")


def build_candidates() -> list[dict]:
    candidates = [
        {
            "name": "Existing configuration baseline",
            "representation": "word_1_2",
            "classifier": "Logistic Regression",
            "parameter_name": "C",
            "parameter": 1.0,
            "baseline": True,
        }
    ]
    for representation in ("word_1_2", "word_1_3", "char_3_5", "word_char"):
        for classifier_name, parameter_name, values in (
            ("Logistic Regression", "C", (0.5, 1.0)),
            ("LinearSVC", "C", (0.5, 1.0)),
            ("MultinomialNB", "alpha", (0.1, 1.0)),
        ):
            for value in values:
                candidates.append(
                    {
                        "name": f"{representation} + {classifier_name} ({parameter_name}={value:g})",
                        "representation": representation,
                        "classifier": classifier_name,
                        "parameter_name": parameter_name,
                        "parameter": value,
                        "baseline": False,
                    }
                )
    return candidates


def model_pipeline(candidate: dict) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", make_transformer(candidate["representation"])),
            (
                "classifier",
                make_classifier(candidate["classifier"], candidate["parameter"], candidate["baseline"]),
            ),
        ]
    )


def score_model(model, texts: pd.Series, labels: pd.Series) -> dict:
    predictions = model.predict(texts)
    report = classification_report(
        labels,
        predictions,
        labels=LABELS,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=LABELS, average="macro", zero_division=0)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
        "confusion_matrix": confusion_matrix(labels, predictions, labels=LABELS).tolist(),
    }


def make_cv_sample(train: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    counts = train["label"].value_counts()
    for label in LABELS:
        size = min(CV_PER_CLASS[label], int(counts[label]))
        pieces.append(train.loc[train["label"].eq(label)].sample(n=size, random_state=RANDOM_STATE))
    return pd.concat(pieces).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def run_cross_validation(cv_frame: pd.DataFrame) -> tuple[list[dict], dict]:
    candidates = build_candidates()
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(
            cv_frame["text"], cv_frame["label"]
        )
    )
    scores = {candidate["name"]: [] for candidate in candidates}
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        fold_train = cv_frame.iloc[train_indices]
        fold_validation = cv_frame.iloc[validation_indices]
        print(f"\nCV fold {fold_number}/5: fitting each feature representation...")
        for representation in ("word_1_2", "word_1_3", "char_3_5", "word_char"):
            train_texts = fold_train["text"]
            validation_texts = fold_validation["text"]
            transformer = make_transformer(representation)
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConvergenceWarning)
                train_matrix = transformer.fit_transform(train_texts)
                validation_matrix = transformer.transform(validation_texts)
            for candidate in candidates:
                if candidate["representation"] != representation:
                    continue
                classifier = make_classifier(
                    candidate["classifier"], candidate["parameter"], candidate["baseline"]
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=ConvergenceWarning)
                    classifier.fit(train_matrix, fold_train["label"])
                fold_prediction = classifier.predict(validation_matrix)
                scores[candidate["name"]].append(
                    float(
                        f1_score(
                            fold_validation["label"],
                            fold_prediction,
                            labels=LABELS,
                            average="macro",
                            zero_division=0,
                        )
                    )
                )
            del transformer, train_matrix, validation_matrix
        print(f"Completed CV fold {fold_number}/5.")

    results = []
    for candidate in candidates:
        fold_scores = scores[candidate["name"]]
        results.append(
            {
                **candidate,
                "cv_macro_f1_mean": float(np.mean(fold_scores)),
                "cv_macro_f1_std": float(np.std(fold_scores, ddof=1)),
                "fold_macro_f1": fold_scores,
                "tfidf_config": vectorizer_config(candidate["representation"]),
            }
        )
    results.sort(key=lambda row: (row["cv_macro_f1_mean"], -row["cv_macro_f1_std"]), reverse=True)
    return results, {"folds": 5, "random_state": RANDOM_STATE, "scoring": "macro_f1"}


def save_joblib(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        joblib.dump(model, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as temp_file:
            json.dump(data, temp_file, indent=2, ensure_ascii=False)
            temp_file.write("\n")
            temp_path = Path(temp_file.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def write_neutral_audit(train: pd.DataFrame, path: Path) -> int:
    neutral = train.loc[train["label"].eq("neutral")].copy()
    neutral["word_count"] = neutral["text"].str.split().str.len()
    neutral["length_band"] = pd.cut(
        neutral["word_count"],
        bins=[-1, 15, 40, np.inf],
        labels=["short (1-15 words)", "medium (16-40 words)", "long (over 40 words)"],
    )
    quotas = {
        "short (1-15 words)": 34,
        "medium (16-40 words)": 33,
        "long (over 40 words)": 33,
    }
    samples = []
    for band, quota in quotas.items():
        group = neutral.loc[neutral["length_band"].eq(band)]
        if len(group) < quota:
            raise ValueError(f"Not enough neutral training reviews in length band {band!r}.")
        samples.append(group.sample(n=quota, random_state=RANDOM_STATE))
    sample = pd.concat(samples).sort_index().copy()
    audit = pd.DataFrame(
        {
            "Text": sample["text"],
            "Original rating (Score)": sample["score"].astype(int),
            "Original sentiment": sample["sentiment"],
            "Original sentiment_label": sample["sentiment_label"],
            "Final sentiment": sample["label"],
            "Word count": sample["word_count"],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(path, index=False, encoding="utf-8-sig")
    return len(audit)


def write_rating_text_disagreement_audit(
    frame: pd.DataFrame, old_model, path: Path
) -> tuple[int, dict[str, int]]:
    predictions = old_model.predict(frame["text"].to_numpy())
    candidates = frame[["text", "score", "sentiment", "sentiment_label", "label"]].copy()
    candidates["Existing model prediction"] = predictions
    candidate_groups = {
        "rating positive, model predicts neutral": candidates.loc[
            candidates["score"].isin([4, 5]) & (candidates["Existing model prediction"] == "neutral")
        ],
        "rating neutral, model predicts positive": candidates.loc[
            (candidates["score"] == 3) & (candidates["Existing model prediction"] == "positive")
        ],
        "rating neutral, model predicts negative": candidates.loc[
            (candidates["score"] == 3) & (candidates["Existing model prediction"] == "negative")
        ],
        "rating negative, model predicts neutral": candidates.loc[
            candidates["score"].isin([1, 2]) & (candidates["Existing model prediction"] == "neutral")
        ],
    }
    selected = []
    counts = {}
    for description, group in candidate_groups.items():
        counts[description] = len(group)
        if not group.empty:
            selected.append(
                group.sample(n=min(10, len(group)), random_state=RANDOM_STATE).assign(
                    **{"Audit category": description}
                )
            )
    if selected:
        result = pd.concat(selected).rename(
            columns={
                "text": "Text",
                "score": "Original rating (Score)",
                "sentiment": "Stored sentiment",
                "sentiment_label": "Stored sentiment_label",
                "label": "Rating-derived sentiment",
            }
        )
    else:
        result = pd.DataFrame(
            columns=[
                "Text",
                "Original rating (Score)",
                "Stored sentiment",
                "Stored sentiment_label",
                "Rating-derived sentiment",
                "Existing model prediction",
                "Audit category",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, encoding="utf-8-sig")
    return len(result), counts


def print_report(title: str, metrics: dict) -> None:
    print(f"\n{title}: accuracy={metrics['accuracy']:.4f}, macro-F1={metrics['macro_f1']:.4f}")
    for label in LABELS:
        row = metrics["per_class"][label]
        print(
            f"  {label:8s} precision={row['precision']:.4f} recall={row['recall']:.4f} "
            f"F1={row['f1']:.4f} support={row['support']}"
        )
    print("  confusion matrix (actual rows; predicted columns; negative, neutral, positive):")
    print(np.asarray(metrics["confusion_matrix"]))


def main() -> None:
    root = Path(__file__).resolve().parent
    old_model_path = root / "sentiment_model.pkl"
    output_dir = root / "model"
    if not old_model_path.exists():
        raise FileNotFoundError(f"Existing model required as comparison baseline: {old_model_path}")

    print("Loading Amazon data and checking the score-derived labels...")
    amazon, label_audit = read_amazon(root / "amazon_reviews_full_clean.csv")
    print("Amazon label audit:", label_audit)
    print("Amazon class counts:")
    print(amazon["label"].value_counts().reindex(LABELS, fill_value=0).to_string())

    train_reviews, amazon_test = train_test_split(
        amazon,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=amazon["label"],
    )
    neutral_count = int(train_reviews["label"].eq("neutral").sum())
    final_per_class_cap = min(60_000, neutral_count * 2)
    final_parts = []
    for label in LABELS:
        group = train_reviews.loc[train_reviews["label"].eq(label)]
        take = len(group) if label == "neutral" else min(final_per_class_cap, len(group))
        final_parts.append(group.sample(n=take, random_state=RANDOM_STATE))
    final_train = pd.concat(final_parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print("\nFinal fit sample class counts (all available Neutral training rows retained):")
    print(final_train["label"].value_counts().reindex(LABELS, fill_value=0).to_string())

    cv_frame = make_cv_sample(train_reviews)
    print("\nAmazon-only CV sample class counts:")
    print(cv_frame["label"].value_counts().reindex(LABELS, fill_value=0).to_string())
    audit_path = output_dir / "neutral_training_audit.csv"
    audit_count = write_neutral_audit(train_reviews, audit_path)
    print(f"Wrote {audit_count} stratified Neutral training examples to {audit_path}")

    old_model = joblib.load(old_model_path)
    old_test_metrics = score_model(old_model, amazon_test["text"], amazon_test["label"])
    disagreement_count, disagreement_directions = write_rating_text_disagreement_audit(
        amazon_test, old_model, output_dir / "rating_text_disagreement_audit.csv"
    )
    print("Old model fixed Amazon holdout metrics:")
    print_report("Old model", old_test_metrics)
    print("Rating/text disagreement candidate counts on held-out Amazon rows:", disagreement_directions)
    print(
        f"Wrote {disagreement_count} real disagreement candidates to "
        f"{output_dir / 'rating_text_disagreement_audit.csv'}; predictions flag candidates for manual review, "
        "not verified label errors."
    )

    cv_results, cv_setup = run_cross_validation(cv_frame)
    baseline = next(row for row in cv_results if row["baseline"])
    winner = cv_results[0]
    improved_on_cv = winner["cv_macro_f1_mean"] > baseline["cv_macro_f1_mean"] + 1e-6
    print("\nCross-validation ranking:")
    for rank, row in enumerate(cv_results, start=1):
        print(
            f"{rank:02d}. {row['name']}: macro-F1={row['cv_macro_f1_mean']:.6f} "
            f"+/- {row['cv_macro_f1_std']:.6f}"
        )
    print(f"\nCV baseline: {baseline['name']} = {baseline['cv_macro_f1_mean']:.6f}")
    print(f"CV winner: {winner['name']} = {winner['cv_macro_f1_mean']:.6f}")

    selected_model = model_pipeline(winner)
    print("\nRefitting the CV-selected configuration on the final Amazon training sample...")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        selected_model.fit(final_train["text"], final_train["label"])

    amazon_metrics = score_model(selected_model, amazon_test["text"], amazon_test["label"])
    print_report("Selected model on untouched Amazon test", amazon_metrics)
    clothing = read_clothing(root / "womens_clothing_reviews.csv")
    clothing_metrics = score_model(selected_model, clothing["text"], clothing["label"])
    print_report("Selected model on external Women's Clothing data", clothing_metrics)
    print("Women's Clothing rating mapping: 1-2 negative, 3 neutral, 4-5 positive")
    print("Women's Clothing class counts:")
    print(clothing["label"].value_counts().reindex(LABELS, fill_value=0).to_string())

    sanity_texts = [
        "The quality is average for the price.",
        "The product is okay, not too good and not too bad.",
        "The product meets my basic expectations.",
        "It is an ordinary product with average quality.",
        "The product is acceptable, but nothing special.",
        "The color is slightly different from the picture.",
        "The product is excellent and I love it.",
        "The product arrived damaged and I am disappointed.",
    ]
    sanity_predictions = selected_model.predict(sanity_texts)
    sanity_probabilities = (
        selected_model.predict_proba(sanity_texts)
        if hasattr(selected_model, "predict_proba")
        else None
    )
    class_positions = {label: index for index, label in enumerate(selected_model.classes_)}
    print("\nEight semantic sanity-check predictions:")
    for index, (text, prediction) in enumerate(zip(sanity_texts, sanity_predictions)):
        line = f"{index + 1}. {text}\n   Predicted: {prediction}"
        if sanity_probabilities is not None:
            probabilities = sanity_probabilities[index]
            line += (
                f"; P(positive)={probabilities[class_positions['positive']]:.6f}"
                f", P(neutral)={probabilities[class_positions['neutral']]:.6f}"
                f", P(negative)={probabilities[class_positions['negative']]:.6f}"
            )
        print(line)

    metadata = {
        "model_name": winner["name"],
        "model_type": winner["classifier"],
        "tfidf_configuration": winner["tfidf_config"],
        "hyperparameters": {
            winner["parameter_name"]: winner["parameter"],
            "class_weight": "balanced" if winner["classifier"] in ("Logistic Regression", "LinearSVC") and not winner["baseline"] else None,
            "fit_prior": False if winner["classifier"] == "MultinomialNB" else None,
            "max_iter": 1500 if winner["classifier"] == "Logistic Regression" else 7000 if winner["classifier"] == "LinearSVC" else None,
        },
        "selection": {
            "criterion": "5-fold stratified cross-validation macro-F1 on Amazon training data only",
            "cv": cv_setup,
            "cv_sample_counts": {
                label: int(count)
                for label, count in cv_frame["label"].value_counts().reindex(LABELS, fill_value=0).items()
            },
            "winner_cv_macro_f1": winner["cv_macro_f1_mean"],
            "winner_cv_macro_f1_std": winner["cv_macro_f1_std"],
            "existing_configuration_cv_macro_f1": baseline["cv_macro_f1_mean"],
            "better_than_existing_configuration_on_same_cv": improved_on_cv,
            "all_candidates": [
                {
                    "name": row["name"],
                    "cv_macro_f1_mean": row["cv_macro_f1_mean"],
                    "cv_macro_f1_std": row["cv_macro_f1_std"],
                    "fold_macro_f1": row["fold_macro_f1"],
                }
                for row in cv_results
            ],
        },
        "amazon_test_metrics": amazon_metrics,
        "existing_model_amazon_test_metrics": old_test_metrics,
        "womens_clothing_metrics": clothing_metrics,
        "label_audit": label_audit,
        "amazon_training_sample_counts": {
            label: int(count)
            for label, count in final_train["label"].value_counts().reindex(LABELS, fill_value=0).items()
        },
        "neutral_training_rows_retained": neutral_count,
        "rating_text_disagreement_candidate_counts": disagreement_directions,
        "womens_clothing_mapping": {"1-2": "negative", "3": "neutral", "4-5": "positive"},
        "sanity_check_predictions": [
            {
                "text": text,
                "predicted": str(prediction),
                "probabilities": (
                    {
                        label: float(sanity_probabilities[index][class_positions[label]])
                        for label in ("positive", "neutral", "negative")
                    }
                    if sanity_probabilities is not None
                    else None
                ),
            }
            for index, (text, prediction) in enumerate(zip(sanity_texts, sanity_predictions))
        ],
        "artifacts": {
            "neutral_training_audit": "neutral_training_audit.csv",
            "rating_text_disagreement_audit": "rating_text_disagreement_audit.csv",
        },
    }
    metadata_path = output_dir / "model_metadata.json"
    save_json(metadata, metadata_path)
    if improved_on_cv:
        model_path = output_dir / "sentiment_model.pkl"
        save_joblib(selected_model, model_path)
        print(f"\nSaved CV-improved model to {model_path}")
    else:
        print(
            "\nThe selected model did not exceed the existing configuration's 5-fold CV macro-F1. "
            "The existing root model was not replaced and no replacement model was written."
        )
    print(f"Saved model comparison and metrics to {metadata_path}")


if __name__ == "__main__":
    main()
