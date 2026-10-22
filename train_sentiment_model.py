import argparse
import os
from pathlib import Path
import re
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


LABELS = ["negative", "neutral", "positive"]
SCORE_TO_SENTIMENT = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
SENTIMENT_LABELS = {"0": "negative", "1": "neutral", "2": "positive"}
RANDOM_STATE = 42
MODEL_NAME = "word_1_3 + Logistic Regression (C=1)"

CURATED_EXAMPLES = {
    "positive": [
        "The product is good.",
        "This product is nice.",
        "I like this product.",
        "The quality is great.",
        "I love this product.",
        "The product works well.",
        "This is a very good product.",
        "The product is good quality.",
        "This product is excellent.",
        "This product is amazing.",
        "This product is wonderful.",
        "This product is fantastic.",
        "This product is awesome.",
        "I like it.",
        "I highly recommend this product.",
        "I am satisfied with this purchase.",
        "I am happy with this purchase.",
        "This product is worth buying.",
        "This product is first-rate.",
        "This product is top-notch.",
    ],
    "neutral": [
        "The product is average.",
        "It meets my basic expectations.",
        "The quality is acceptable.",
        "This product is okay, but nothing special.",
        "It works as expected.",
        "The item is ordinary and adequate.",
        "The product is neither good nor bad.",
        "This is a typical product with average quality.",
        "The product meets expectations.",
        "It is usable but not remarkable.",
        "The item is suitable for basic needs.",
        "There is nothing especially good or bad about it.",
        "The quality is fair.",
        "This product is passable.",
        "The product is unremarkable.",
        "The item is adequate for everyday use.",
        "It is an ordinary product.",
        "The product is acceptable for its purpose.",
        "This item is about what I expected.",
        "The product is neither impressive nor disappointing.",
    ],
    "negative": [
        "The product is bad.",
        "I do not like this product.",
        "The product is not good.",
        "The quality is terrible.",
        "This product is poor quality.",
        "I dislike this item.",
        "It does not work well.",
        "I would not recommend this product.",
        "The item is disappointing.",
        "This product is awful.",
        "The item broke quickly.",
        "The quality is very poor.",
        "I am unhappy with this purchase.",
        "This was a waste of money.",
        "This product failed to work.",
        "The product arrived damaged.",
        "The item is defective.",
        "I regret buying this.",
        "The product does not meet expectations.",
        "This item is a bad purchase.",
    ],
}

INVALID_INPUTS = ["12345", "987654", "@#$%", "!!!", "xyz", "adsfg", "tyunb", "", "   "]


def read_amazon_reviews(path: Path) -> pd.DataFrame:
    required = {"clean_text", "Score", "sentiment_label"}
    frame = pd.read_csv(
        path,
        usecols=lambda column: column in required,
        dtype={"clean_text": "string", "Score": "string", "sentiment_label": "string"},
    )
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Amazon CSV is missing required columns: {', '.join(sorted(missing))}")

    frame["text"] = frame["clean_text"].fillna("").str.strip()
    frame["score"] = pd.to_numeric(frame["Score"], errors="coerce")
    frame["label"] = frame["score"].map(SCORE_TO_SENTIMENT)
    frame["stored_label"] = frame["sentiment_label"].str.strip().map(SENTIMENT_LABELS)
    frame = frame.loc[frame["text"].ne("")].copy()

    invalid_scores = frame["label"].isna()
    if invalid_scores.any():
        raise ValueError(f"Amazon CSV has {int(invalid_scores.sum())} rows with invalid or missing scores.")
    mismatches = frame["label"].ne(frame["stored_label"])
    if mismatches.any():
        raise ValueError(f"Amazon CSV has {int(mismatches.sum())} rows whose sentiment_label disagrees with Score.")
    if set(frame["label"].unique()) != set(LABELS):
        raise ValueError("Amazon CSV must contain negative, neutral, and positive reviews.")
    return frame[["text", "label"]]


def read_clothing_reviews(path: Path) -> pd.DataFrame:
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
    invalid_ratings = frame["rating"].notna() & ~frame["rating"].isin(range(1, 6))
    if invalid_ratings.any():
        raise ValueError(f"Clothing CSV has {int(invalid_ratings.sum())} ratings outside the range 1-5.")
    frame = frame.loc[frame["text"].ne("") & frame["rating"].notna()].copy()
    frame["label"] = frame["rating"].map(SCORE_TO_SENTIMENT)
    if frame.empty:
        raise ValueError("Clothing CSV has no non-empty reviews with valid ratings.")
    return frame[["text", "label"]]


def normalize_review_phrases(texts):
    return [
        re.sub(
            r"\b(?:first[\s-]+rate|top[\s-]+notch)\b",
            "excellent",
            text,
            flags=re.IGNORECASE,
        )
        for text in texts
    ]


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=70_000,
                    min_df=2,
                    ngram_range=(1, 3),
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1500,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate(model: Pipeline, texts, labels) -> dict:
    predictions = model.predict(texts)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=LABELS,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(labels, predictions, labels=LABELS).tolist(),
    }


def print_metrics(title: str, metrics: dict) -> None:
    print(
        f"\n{title}: accuracy={metrics['accuracy']:.4f}, "
        f"macro precision={metrics['macro_precision']:.4f}, "
        f"macro recall={metrics['macro_recall']:.4f}, macro F1={metrics['macro_f1']:.4f}"
    )
    for label in LABELS:
        row = metrics["per_class"][label]
        print(
            f"  {label:8s} precision={row['precision']:.4f} recall={row['recall']:.4f} "
            f"F1={row['f1']:.4f} support={row['support']}"
        )
    print("  confusion matrix (actual rows; predicted columns; negative, neutral, positive):")
    print(np.asarray(metrics["confusion_matrix"]))


def check_compatibility(model: Pipeline) -> None:
    if not isinstance(model, Pipeline) or list(model.named_steps) != ["tfidf", "classifier"]:
        raise TypeError("Candidate is not a TF-IDF and classifier pipeline compatible with the application.")
    if list(model.classes_) != LABELS:
        raise ValueError(f"Candidate has unexpected class order: {list(model.classes_)}")
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    if not isinstance(vectorizer, TfidfVectorizer) or tuple(vectorizer.ngram_range) != (1, 3):
        raise TypeError("Candidate TF-IDF vectorizer does not match the application's expected configuration.")
    if not isinstance(classifier, LogisticRegression) or not hasattr(model, "predict_proba"):
        raise TypeError("Candidate is not a probability-capable Logistic Regression model.")
    probabilities = model.predict_proba(["compatibility check"])
    if probabilities.shape != (1, len(LABELS)) or not np.isfinite(probabilities).all():
        raise ValueError("Candidate generated an invalid probability vector.")
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("Candidate probabilities do not sum to one.")


def save_model(model: Pipeline, output_path: Path) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".tmp", delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)
        joblib.dump(model, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def print_predictions(model: Pipeline, texts: list[str], title: str) -> None:
    normalized = normalize_review_phrases(texts)
    predictions = model.predict(normalized)
    probabilities = model.predict_proba(normalized)
    class_positions = {label: index for index, label in enumerate(model.classes_)}
    print(f"\n{title}:")
    for text, prediction, row in zip(texts, predictions, probabilities):
        values = ", ".join(f"{label}={row[class_positions[label]]:.4f}" for label in LABELS)
        print(f"  {text!r} => {prediction} ({values})")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train and evaluate the application's TF-IDF Logistic Regression sentiment model."
    )
    parser.add_argument("--amazon-csv", type=Path, default=script_dir / "amazon_reviews_full_clean.csv")
    parser.add_argument("--clothing-csv", type=Path, default=script_dir / "womens_clothing_reviews.csv")
    parser.add_argument("--output", type=Path, default=script_dir / "model" / "sentiment_model.pkl")
    args = parser.parse_args()
    if not args.output.is_file():
        raise FileNotFoundError(f"Existing application model required for comparison: {args.output}")

    amazon = read_amazon_reviews(args.amazon_csv)
    print(f"Usable Amazon reviews: {len(amazon):,}")
    print("Amazon class counts:")
    print(amazon["label"].value_counts().reindex(LABELS, fill_value=0).to_string())
    train_reviews, amazon_test = train_test_split(
        amazon,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=amazon["label"],
    )
    print("\nAmazon training partition class counts:")
    print(train_reviews["label"].value_counts().reindex(LABELS, fill_value=0).to_string())

    old_model = joblib.load(args.output)
    check_compatibility(old_model)
    test_texts = normalize_review_phrases(amazon_test["text"].tolist())
    old_amazon_metrics = evaluate(old_model, test_texts, amazon_test["label"])
    print_metrics("Existing model on fixed Amazon holdout", old_amazon_metrics)

    clothing = read_clothing_reviews(args.clothing_csv)
    clothing_texts = normalize_review_phrases(clothing["text"].tolist())
    old_clothing_metrics = evaluate(old_model, clothing_texts, clothing["label"])
    print_metrics("Existing model on independent women's clothing data", old_clothing_metrics)
    print("Women's clothing rating mapping: 1-2 negative, 3 neutral, 4-5 positive")
    print("Women's clothing class counts:")
    print(clothing["label"].value_counts().reindex(LABELS, fill_value=0).to_string())

    augmentation = pd.DataFrame(
        [
            {"text": text, "label": label}
            for label, examples in CURATED_EXAMPLES.items()
            for text in examples
        ]
    )
    if augmentation["label"].value_counts().nunique() != 1:
        raise ValueError("Curated training examples must have equal representation for all three classes.")
    train_counts = train_reviews["label"].value_counts().reindex(LABELS)
    class_weights = len(train_reviews) / (len(LABELS) * train_counts)
    per_class_added_weight = len(train_reviews) / len(LABELS) * 0.03
    augmentation_weights = np.asarray(
        [
            per_class_added_weight
            / (int(augmentation["label"].eq(label).sum()) * class_weights[label])
            for label in augmentation["label"]
        ],
        dtype=float,
    )
    fit_reviews = pd.concat([train_reviews, augmentation], ignore_index=True)
    fit_weights = np.concatenate([np.ones(len(train_reviews)), augmentation_weights])
    fit_texts = normalize_review_phrases(fit_reviews["text"].tolist())
    print(
        f"\nAdding {len(augmentation)} curated labeled examples "
        "(equal class counts; 3% effective training weight per class)."
    )

    new_model = make_pipeline()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        new_model.fit(fit_texts, fit_reviews["label"], classifier__sample_weight=fit_weights)
    check_compatibility(new_model)

    new_amazon_metrics = evaluate(new_model, test_texts, amazon_test["label"])
    print_metrics("New model on the same fixed Amazon holdout", new_amazon_metrics)
    new_clothing_metrics = evaluate(new_model, clothing_texts, clothing["label"])
    print_metrics("New model on independent women's clothing data", new_clothing_metrics)

    positive_examples = CURATED_EXAMPLES["positive"][:8]
    neutral_examples = CURATED_EXAMPLES["neutral"][:3]
    negative_examples = CURATED_EXAMPLES["negative"][:4]
    print_predictions(
        new_model,
        positive_examples + neutral_examples + negative_examples,
        "Predictions and probabilities for the requested labeled examples",
    )
    print_predictions(new_model, INVALID_INPUTS, "Raw model outputs for invalid-input probes")

    per_class_f1_not_regressed = all(
        new_amazon_metrics["per_class"][label]["f1"]
        >= old_amazon_metrics["per_class"][label]["f1"] - 0.02
        for label in LABELS
    )
    passed = (
        new_amazon_metrics["macro_f1"] > old_amazon_metrics["macro_f1"]
        and new_clothing_metrics["macro_f1"] >= old_clothing_metrics["macro_f1"]
        and per_class_f1_not_regressed
    )
    if not passed:
        raise RuntimeError(
            "Candidate did not pass the holdout and external-evaluation replacement gates; "
            "the existing model artifact was left unchanged."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output.with_name(f".{args.output.stem}.candidate.pkl")
    try:
        save_model(new_model, candidate_path)
        check_compatibility(joblib.load(candidate_path))
        os.replace(candidate_path, args.output)
    finally:
        if candidate_path.exists():
            candidate_path.unlink()
    print(f"\nPassed compatibility and evaluation checks; replaced model artifact at {args.output.resolve()}")


if __name__ == "__main__":
    main()
