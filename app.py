from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import json
import os
from pathlib import Path
import re

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parent
LOGIN_PAGE_PATH = ROOT / "assets" / "login_page (3).py"
LOGIN_PAGE_SPEC = spec_from_file_location("assets.login_page", LOGIN_PAGE_PATH)
if LOGIN_PAGE_SPEC is None or LOGIN_PAGE_SPEC.loader is None:
    raise ImportError(f"Could not load the login page module at {LOGIN_PAGE_PATH}")
LOGIN_PAGE_MODULE = module_from_spec(LOGIN_PAGE_SPEC)
LOGIN_PAGE_SPEC.loader.exec_module(LOGIN_PAGE_MODULE)
show_login_page = LOGIN_PAGE_MODULE.show_login_page
MODEL_PATH = ROOT / "model" / "sentiment_model.pkl"
METADATA_PATH = ROOT / "model" / "model_metadata.json"
CLOTHING_PATH = ROOT / "womens_clothing_reviews.csv"
LABELS = ["negative", "neutral", "positive"]
DISPLAY_LABELS = ["Negative", "Neutral", "Positive"]
RATING_TO_SENTIMENT = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
SENTIMENT_LABELS = {
    "negative": "negative",
    "neg": "negative",
    "0": "negative",
    "neutral": "neutral",
    "neu": "neutral",
    "1": "neutral",
    "positive": "positive",
    "pos": "positive",
    "2": "positive",
}
TEXT_COLUMN_PREFERENCE = ("review text", "clean_text", "text", "review", "content", "summary")


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Latest model file not found: {MODEL_PATH}")
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Model metadata not found: {METADATA_PATH}")

    with METADATA_PATH.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "predict"):
        raise TypeError(f"The saved artifact at {MODEL_PATH} does not provide predict().")
    if list(model.classes_) != LABELS:
        raise ValueError(f"Unexpected model classes in {MODEL_PATH}: {list(model.classes_)}")

    expected_name = metadata.get("model_name")
    classifier = model.named_steps.get("classifier")
    transformer = model.named_steps.get("tfidf")
    if expected_name != "word_1_3 + Logistic Regression (C=1)":
        raise ValueError(f"Unexpected latest-model metadata name: {expected_name!r}")
    if classifier is None or classifier.__class__.__name__ != "LogisticRegression":
        raise TypeError("The latest model is not the Logistic Regression model recorded in metadata.")
    if transformer is None or transformer.__class__.__name__ != "TfidfVectorizer":
        raise TypeError("The latest model is missing its expected TF-IDF vectorizer.")
    if tuple(transformer.ngram_range) != (1, 3):
        raise ValueError(f"Model TF-IDF n-gram range does not match metadata: {transformer.ngram_range}")
    if not hasattr(model, "predict_proba"):
        raise TypeError("The latest Logistic Regression model does not provide predict_proba().")
    return model, metadata


def display_sentiment(label: str) -> str:
    if pd.isna(label):
        return ""
    return str(label).capitalize()


def sentiment_banner(label: str) -> None:
    if label == "positive":
        st.success("🟢 POSITIVE")
    elif label == "neutral":
        st.warning("🟡 NEUTRAL")
    elif label == "negative":
        st.error("🔴 NEGATIVE")
    else:
        raise ValueError(f"Model returned an unsupported sentiment: {label}")


def predict_one(model, text: str) -> tuple[str, dict[str, float] | None]:
    predicted = str(model.predict([text])[0])
    if predicted not in LABELS:
        raise ValueError(f"Model returned an unsupported sentiment: {predicted}")
    probabilities = None
    if hasattr(model, "predict_proba"):
        values = model.predict_proba([text])[0]
        probabilities = {label: float(values[list(model.classes_).index(label)]) for label in LABELS}
    return predicted, probabilities


def has_meaningful_review_text(text: str, model) -> bool:
    words = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
    meaningful_words = [word for word in words if len(word) >= 2]
    if not meaningful_words or sum(len(word) for word in meaningful_words) < 3:
        return False
    vectorizer = model.named_steps.get("tfidf")
    if vectorizer is None:
        raise TypeError("The loaded sentiment model is missing its TF-IDF vectorizer.")
    vocabulary = vectorizer.vocabulary_
    return any(word.lower() in vocabulary for word in meaningful_words)


def normalize_review_phrases(text: str) -> str:
    return re.sub(
        r"\b(?:first[\s-]+rate|top[\s-]+notch)\b",
        "excellent",
        text,
        flags=re.IGNORECASE,
    )


def probability_metrics(metrics: dict) -> dict[str, float]:
    per_class = metrics["per_class"]
    total_support = sum(per_class[label]["support"] for label in LABELS)
    if not total_support:
        raise ValueError("Saved performance metrics do not contain class support.")
    return {
        metric: sum(
            per_class[label][metric] * per_class[label]["support"] for label in LABELS
        )
        / total_support
        for metric in ("precision", "recall", "f1")
    }


def render_home() -> None:
    st.markdown(
        f'<div class="hero"><div class="hero-kicker">🤖 {PROJECT_TITLE}</div>'
        f'<div class="hero-title">{PROJECT_TITLE}</div>'
        '<div class="hero-copy"><strong>Transforming E-Commerce Reviews into Actionable Insights</strong><br>'
        'Understand what customers really think. Analyze individual reviews or thousands of e-commerce reviews '
        'and automatically classify them as Positive, Neutral, or Negative.</div></div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(2)
    columns[0].markdown(
        '<div class="info-card"><div class="info-title">✍️ Analyze a Review</div>'
        '<div class="info-copy">Analyze one customer review instantly.</div></div>',
        unsafe_allow_html=True,
    )
    if columns[0].button("Open Single Review", key="home_single_review", use_container_width=True):
        set_current_page("✍️ Single Review")
        st.rerun()
    columns[1].markdown(
        '<div class="info-card"><div class="info-title">📊 Analyze Dataset</div>'
        '<div class="info-copy">Upload a CSV and analyze multiple reviews.</div></div>',
        unsafe_allow_html=True,
    )
    if columns[1].button("Open Dataset Analysis", key="home_dataset_analysis", use_container_width=True):
        set_current_page("📊 Dataset Analysis")
        st.rerun()

    st.markdown('<div class="section-label">SENTIMENT AT A GLANCE</div>', unsafe_allow_html=True)
    columns = st.columns(3)
    cards = [
        ("🟢", "POSITIVE", "Happy customer opinions", "sentiment-positive"),
        ("🟡", "NEUTRAL", "Balanced or average opinions", "sentiment-neutral"),
        ("🔴", "NEGATIVE", "Unhappy customer opinions", "sentiment-negative"),
    ]
    for column, (icon, title, description, css_class) in zip(columns, cards):
        column.markdown(
            f'<div class="sentiment-card {css_class}"><div class="sentiment-icon">{icon}</div>'
            f'<div class="sentiment-name">{title}</div><div class="sentiment-desc">{description}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">HOW IT WORKS</div>', unsafe_allow_html=True)
    steps = ["Customer Review", "Text Processing", "TF-IDF", "Sentiment Analysis", "Positive / Neutral / Negative"]
    pipeline = '<div class="pipeline">'
    for index, step in enumerate(steps):
        pipeline += f'<div class="pipeline-step">{step}</div>'
        if index < len(steps) - 1:
            pipeline += '<div class="pipeline-arrow">→</div>'
    st.markdown(pipeline + "</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-label">CORE CAPABILITIES</div>', unsafe_allow_html=True)
    columns = st.columns(4)
    features = [
        ("⚡", "Real-Time Analysis", "Analyze reviews instantly."),
        ("📂", "CSV Analysis", "Upload review datasets for bulk analysis."),
        ("🛒", "E-Commerce Reviews", "Designed for customer review analysis."),
        ("📈", "Sentiment Insights", "Understand positive, neutral, and negative opinions."),
    ]
    for column, (icon, title, description) in zip(columns, features):
        column.markdown(
            f'<div class="feature-card"><div class="feature-icon">{icon}</div><div class="feature-title">{title}</div>'
            f'<div class="feature-copy">{description}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">PROJECT FLOW</div>', unsafe_allow_html=True)
    columns = st.columns(4)
    for index, title in enumerate(["Review Collection", "Data Processing", "Sentiment Prediction", "Visual Insights"], 1):
        columns[index - 1].markdown(
            f'<div class="flow-card"><div class="flow-number">0{index}</div><div class="flow-title">{title}</div></div>',
            unsafe_allow_html=True,
        )


def render_single_review(model) -> None:
    st.markdown(
        '<div class="page-eyebrow">✍️ CUSTOMER VOICE</div>'
        '<h1 class="page-title">Analyze Customer Review</h1>'
        '<p class="page-subtitle">Turn one customer opinion into an actionable sentiment insight.</p>',
        unsafe_allow_html=True,
    )
    with st.form("single_review_form"):
        review = st.text_area(
            "Enter a customer review...",
            height=190,
            key="single_review_text",
            placeholder="Example: The product quality is excellent and I really love it.",
        )
        submitted = st.form_submit_button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

    if not submitted:
        return
    text = review.strip()
    if not text:
        st.error("Enter review text before analyzing.")
        return

    normalized_text = normalize_review_phrases(text)
    if not has_meaningful_review_text(normalized_text, model):
        st.warning(
            "Please enter meaningful review text using words recognized by the sentiment model."
        )
        return
    predicted, probabilities = predict_one(model, normalized_text)
    st.session_state.setdefault("history", []).append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "review": text,
            "predicted_sentiment": display_sentiment(predicted),
            "positive_probability": probabilities["positive"] if probabilities is not None else None,
            "neutral_probability": probabilities["neutral"] if probabilities is not None else None,
            "negative_probability": probabilities["negative"] if probabilities is not None else None,
        }
    )
    messages = {
        "positive": ("🟢", "POSITIVE", "Customer sentiment is positive.", "result-positive"),
        "neutral": ("🟡", "NEUTRAL", "Customer sentiment is neutral.", "result-neutral"),
        "negative": ("🔴", "NEGATIVE", "Customer sentiment is negative.", "result-negative"),
    }
    icon, title, message, css_class = messages[predicted]
    st.markdown(
        f'<div class="result-card {css_class}"><div class="result-icon">{icon}</div>'
        f'<div><div class="result-title">{title}</div><div class="result-message">{message}</div></div></div>',
        unsafe_allow_html=True,
    )
    if probabilities is not None:
        st.markdown('<div class="section-label">SENTIMENT SIGNAL</div>', unsafe_allow_html=True)
        columns = st.columns(3)
        columns[0].metric("Positive probability", f"{probabilities['positive']:.2%}")
        columns[1].metric("Neutral probability", f"{probabilities['neutral']:.2%}")
        columns[2].metric("Negative probability", f"{probabilities['negative']:.2%}")


def text_column_options(frame: pd.DataFrame) -> list[str]:
    text_columns = [
        column
        for column in frame.columns
        if pd.api.types.is_object_dtype(frame[column])
        or pd.api.types.is_string_dtype(frame[column])
    ]
    if not text_columns:
        return []
    lower_names = {str(column).strip().lower(): column for column in text_columns}
    preferred = [lower_names[name] for name in TEXT_COLUMN_PREFERENCE if name in lower_names]
    return preferred + [column for column in text_columns if column not in preferred]


def guess_label_column(frame: pd.DataFrame, text_column: str) -> str | None:
    priorities = ("sentiment", "sentiment_label", "label", "rating", "score", "stars")
    names = {str(column).strip().lower(): column for column in frame.columns if column != text_column}
    for name in priorities:
        if name in names:
            return names[name]
    return None


def map_ground_truth(values: pd.Series, column_name: str) -> tuple[pd.Series, int]:
    name = str(column_name).strip().lower()
    normalized = values.astype("string").str.strip().str.lower()
    numeric = pd.to_numeric(values, errors="coerce")

    if name in {"rating", "score", "stars", "star_rating"}:
        mapped = numeric.map(RATING_TO_SENTIMENT)
    elif name == "sentiment_label":
        mapped = normalized.map(SENTIMENT_LABELS)
    else:
        mapped = normalized.map(SENTIMENT_LABELS)
        numeric_label_values = set(numeric.dropna().unique())
        if name == "label" and numeric_label_values.issubset({0, 1, 2}) and numeric_label_values:
            mapped = numeric.map({0: "negative", 1: "neutral", 2: "positive"})

    mapped = mapped.astype("string")
    invalid_count = int(mapped.isna().sum())
    return mapped, invalid_count


def render_evaluation_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    actual_values = actual.astype(str).to_numpy()
    predicted_values = predicted.astype(str).to_numpy()
    report = classification_report(
        actual_values,
        predicted_values,
        labels=LABELS,
        target_names=DISPLAY_LABELS,
        output_dict=True,
        zero_division=0,
    )
    accuracy = accuracy_score(actual_values, predicted_values)
    st.subheader("Evaluation metrics")
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{accuracy:.4f}")
    cols[1].metric("Macro Precision", f"{report['macro avg']['precision']:.4f}")
    cols[2].metric("Macro Recall", f"{report['macro avg']['recall']:.4f}")
    cols[3].metric("Macro F1", f"{report['macro avg']['f1-score']:.4f}")
    weighted_cols = st.columns(3)
    weighted_cols[0].metric("Weighted Precision", f"{report['weighted avg']['precision']:.4f}")
    weighted_cols[1].metric("Weighted Recall", f"{report['weighted avg']['recall']:.4f}")
    weighted_cols[2].metric("Weighted F1", f"{report['weighted avg']['f1-score']:.4f}")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Sentiment": title,
                    "Precision": report[title]["precision"],
                    "Recall": report[title]["recall"],
                    "F1": report[title]["f1-score"],
                    "Support": int(report[title]["support"]),
                }
                for title in DISPLAY_LABELS
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    matrix = confusion_matrix(actual_values, predicted_values, labels=LABELS)
    st.caption("Confusion matrix: rows are actual labels; columns are predicted labels.")
    st.dataframe(
        pd.DataFrame(matrix, index=DISPLAY_LABELS, columns=DISPLAY_LABELS),
        width="stretch",
    )
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "confusion_matrix": matrix.tolist(),
    }


def render_dataset_analysis(model) -> None:
    st.markdown(
        '<div class="page-eyebrow">📊 REVIEW INTELLIGENCE</div>'
        '<h1 class="page-title">Dataset Analysis</h1>'
        '<p class="page-subtitle">Analyze customer feedback at scale and surface the sentiment behind every review.</p>',
        unsafe_allow_html=True,
    )
    analysis_mode = st.radio(
        "Choose a dataset",
        ["Upload CSV", "Women's Clothing dataset"],
        horizontal=True,
        key="dataset_analysis_mode",
    )

    if analysis_mode == "Women's Clothing dataset":
        if not CLOTHING_PATH.is_file():
            st.error(f"Women's Clothing dataset not found: {CLOTHING_PATH}")
            return
        frame = pd.read_csv(CLOTHING_PATH, usecols=["Review Text", "Rating"])
        text_column = "Review Text"
        actual, invalid_count = map_ground_truth(frame["Rating"], "Rating")
        valid = frame["Review Text"].fillna("").astype(str).str.strip().ne("") & actual.notna()
        texts = frame.loc[valid, "Review Text"].fillna("").astype(str).str.strip()
        actual = actual.loc[valid].astype(str)
        st.caption(f"Ratings: 1–2 Negative, 3 Neutral, 4–5 Positive • {len(texts):,} non-empty reviews ready for analysis.")
        if invalid_count:
            st.warning(f"{invalid_count:,} rows had missing or invalid ratings.")
        if texts.empty:
            st.error("No non-empty reviews with valid ratings are available.")
            return
        predictions = pd.Series(model.predict(texts.to_numpy()), index=texts.index)
        render_dataset_summary(predictions)
        result = pd.DataFrame(
            {
                "Review": texts,
                "Actual": actual.map(display_sentiment),
                "Predicted Sentiment": predictions.map(display_sentiment),
            }
        )
        render_evaluation_metrics(actual, predictions)
        st.markdown('<div class="section-label">REVIEW PREDICTIONS</div>', unsafe_allow_html=True)
        st.dataframe(result, hide_index=True, width="stretch")
        return

    st.markdown('<div class="upload-panel">📁 <strong>Upload Review Dataset</strong><br><span>CSV files with a review text column are supported.</span></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="dataset_csv_upload", label_visibility="collapsed")
    if uploaded is None:
        st.info("Upload a CSV to predict sentiment for its reviews.")
        return
    try:
        frame = pd.read_csv(BytesIO(uploaded.getvalue()))
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        return
    if frame.empty:
        st.error("The uploaded CSV contains no rows.")
        return

    text_options = text_column_options(frame)
    if not text_options:
        st.error("No text column was detected. Include a text column such as Review Text or clean_text.")
        return
    guessed_text = text_options[0]
    text_column = st.selectbox(
        "Review text column",
        text_options,
        index=text_options.index(guessed_text),
        key="dataset_text_column",
    )
    label_options = ["No ground-truth labels"] + [column for column in frame.columns if column != text_column]
    guessed_label = guess_label_column(frame, text_column)
    label_index = label_options.index(guessed_label) if guessed_label in label_options else 0
    label_column = st.selectbox(
        "Actual sentiment/rating column",
        label_options,
        index=label_index,
        key="dataset_label_column",
    )

    texts = frame[text_column].fillna("").astype(str).str.strip()
    nonempty = texts.ne("")
    predictions = pd.Series(index=frame.index, dtype="string")
    if nonempty.any():
        predictions.loc[nonempty] = model.predict(texts.loc[nonempty].to_numpy())
        render_dataset_summary(predictions.loc[nonempty])
    result = pd.DataFrame(
        {
            "Review": texts,
            "Predicted Sentiment": predictions.map(display_sentiment),
        }
    )
    if label_column != "No ground-truth labels":
        actual, invalid_count = map_ground_truth(frame[label_column], str(label_column))
        valid_actual = nonempty & actual.notna()
        result["Actual"] = actual.map(display_sentiment)
        if invalid_count:
            st.warning(
                f"{invalid_count:,} rows in {label_column!r} could not be mapped to a known sentiment "
                "and are excluded from metrics."
            )
        if valid_actual.any():
            render_evaluation_metrics(actual.loc[valid_actual], predictions.loc[valid_actual])
        else:
            st.info("Ground-truth labels are unavailable or unrecognized; accuracy and other metrics were not calculated.")
    else:
        st.info("No ground-truth column selected; metrics are not calculated.")
    st.caption(f"Predicted {int(nonempty.sum()):,} non-empty reviews; skipped {int((~nonempty).sum()):,} empty reviews.")
    st.markdown('<div class="section-label">REVIEW PREDICTIONS</div>', unsafe_allow_html=True)
    st.dataframe(result, hide_index=True, width="stretch")


def render_dataset_summary(predictions: pd.Series) -> None:
    counts = predictions.astype("string").value_counts()
    cards = [
        ("Reviews Analyzed", len(predictions), "summary-neutral"),
        ("Sentiments Detected", int(predictions.nunique()), "summary-neutral"),
        ("Positive Reviews", int(counts.get("positive", 0)), "summary-positive"),
        ("Neutral Reviews", int(counts.get("neutral", 0)), "summary-neutral"),
        ("Negative Reviews", int(counts.get("negative", 0)), "summary-negative"),
    ]
    columns = st.columns(5)
    for column, (label, value, css_class) in zip(columns, cards):
        column.markdown(
            f'<div class="summary-card {css_class}"><div class="summary-value">{value:,}</div>'
            f'<div class="summary-label">{label}</div></div>',
            unsafe_allow_html=True,
        )
    chart = pd.DataFrame(
        {"Sentiment": ["Positive", "Neutral", "Negative"], "Reviews": [counts.get("positive", 0), counts.get("neutral", 0), counts.get("negative", 0)]}
    ).set_index("Sentiment")
    st.markdown('<div class="section-label">SENTIMENT DISTRIBUTION</div>', unsafe_allow_html=True)
    st.bar_chart(chart, color="#64d59b", height=250)


def render_model_performance(metadata: dict) -> None:
    st.header("Model Performance")
    st.subheader(metadata["model_name"])
    st.caption(f"Metrics from saved metadata: {METADATA_PATH}")

    sections = [
        ("Amazon test set", metadata["amazon_test_metrics"]),
        ("Women's Clothing external evaluation", metadata["womens_clothing_metrics"]),
    ]
    for title, metrics in sections:
        st.subheader(title)
        weighted = probability_metrics(metrics)
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{metrics['accuracy']:.4f}")
        cols[1].metric("Macro F1", f"{metrics['macro_f1']:.4f}")
        cols[2].metric("Weighted F1", f"{metrics['weighted_f1']:.4f}")
        cols[3].metric("Macro Precision", f"{metrics['macro_precision']:.4f}")
        cols = st.columns(3)
        cols[0].metric("Macro Recall", f"{metrics['macro_recall']:.4f}")
        cols[1].metric("Weighted Precision", f"{weighted['precision']:.4f}")
        cols[2].metric("Weighted Recall", f"{weighted['recall']:.4f}")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Sentiment": display_sentiment(label),
                        "Precision": metrics["per_class"][label]["precision"],
                        "Recall": metrics["per_class"][label]["recall"],
                        "F1": metrics["per_class"][label]["f1"],
                        "Support": metrics["per_class"][label]["support"],
                    }
                    for label in LABELS
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    selection = metadata["selection"]
    st.caption(
        f"5-fold stratified CV macro-F1: {selection['winner_cv_macro_f1']:.4f} "
        f"(± {selection['winner_cv_macro_f1_std']:.4f})."
    )


def render_history() -> None:
    st.markdown(
        '<div class="page-eyebrow">🕘 ACTIVITY LOG</div>'
        '<h1 class="page-title">Analysis History</h1>'
        '<p class="page-subtitle">A record of the customer reviews analyzed in this session.</p>',
        unsafe_allow_html=True,
    )
    history = st.session_state.get("history", [])
    if not history:
        st.info("Single-review analyses will appear here during this session.")
        return
    frame = pd.DataFrame(history)
    frame = frame.rename(
        columns={"review": "Review", "predicted_sentiment": "Prediction", "timestamp": "Timestamp"}
    )
    frame["Prediction"] = frame["Prediction"].map(
        lambda value: {"Positive": "🟢 Positive", "Neutral": "🟡 Neutral", "Negative": "🔴 Negative"}.get(value, value)
    )
    frame = frame[["Review", "Prediction", "Timestamp"]]
    st.dataframe(frame, hide_index=True, width="stretch")
    st.download_button(
        "Download history CSV",
        frame.to_csv(index=False).encode("utf-8"),
        file_name="sentiment_history.csv",
        mime="text/csv",
    )


def render_about(metadata: dict, model) -> None:
    st.markdown(
        '<div class="page-eyebrow">ℹ️ PROJECT OVERVIEW</div>'
        '<h1 class="page-title">About the Project</h1>'
        '<p class="page-subtitle">A practical intelligence layer for understanding e-commerce customer voice.</p>',
        unsafe_allow_html=True,
    )
    sections = [
        ("🎯 Project Objective", "Convert customer reviews into clear sentiment insights that help teams understand satisfaction at a glance."),
        ("🔍 Problem Statement", "Manually reading large volumes of reviews is slow, inconsistent, and makes it difficult to spot broad customer trends."),
        ("🧠 Solution", "This application analyzes individual reviews and uploaded datasets, classifying each opinion as Positive, Neutral, or Negative."),
        ("🛠️ Technologies Used", "Python • Pandas • Scikit-learn • NLTK • TF-IDF • Logistic Regression • Streamlit • Plotly"),
        ("📊 Sentiment Classes", "🟢 Positive   •   🟡 Neutral   •   🔴 Negative"),
        ("🌐 Applications", "E-commerce • Customer feedback • Product analysis • Business insights"),
        ("🚀 Future Scope", "Real-time review collection • More e-commerce sources • Aspect-based sentiment analysis • Advanced NLP models"),
    ]
    for index in range(0, len(sections), 2):
        columns = st.columns(2)
        for column, (heading, body) in zip(columns, sections[index:index + 2]):
            column.markdown(
                f'<div class="about-card"><div class="about-heading">{heading}</div>'
                f'<div class="about-body">{body}</div></div>',
                unsafe_allow_html=True,
            )


PAGE_OPTIONS = ["🏠 Home", "✍️ Single Review", "📊 Dataset Analysis", "🕘 History", "ℹ️ About"]
PROJECT_TITLE = "REAL-TIME SENTIMENT ANALYSIS WITH WEB SCRAPING"


def set_current_page(page: str) -> None:
    normalized_page = page if page in PAGE_OPTIONS else "🏠 Home"
    st.session_state["selected_page"] = normalized_page
    st.query_params["page"] = normalized_page


def resolve_current_page() -> str:
    query_page = st.query_params.get("page", "🏠 Home")
    if query_page in PAGE_OPTIONS:
        st.session_state["selected_page"] = query_page
        return query_page
    st.session_state["selected_page"] = "🏠 Home"
    st.query_params["page"] = "🏠 Home"
    return "🏠 Home"


def _new_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root { --bg:#09111f; --panel:#111d31; --panel2:#172943; --text:#f4f7fb; --muted:#91a0b7; --line:#294362; --blue:#76a9ff; --purple:#a98cff; --green:#64d59b; --yellow:#f4c95d; --red:#f47d83; }
        .stApp { background:radial-gradient(circle at 78% -10%,#243b69 0,transparent 34%),var(--bg); color:var(--text); font-family:'DM Sans',sans-serif; }
        [data-testid="stSidebar"] { background:linear-gradient(180deg,#111d31,#0b1322); border-right:1px solid #233652; }
        [data-testid="stSidebar"] * { font-family:'DM Sans',sans-serif; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.55rem .65rem; border-radius:10px; transition:background .15s ease,color .15s ease; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:#1c2b43; color:#fff; }
        [data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"] [data-testid="stIconMaterial"],
        [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] { display:none!important; }
        [data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"]::after,
        [data-testid="stExpandSidebarButton"]::after { color:#7edcff; font:600 1.35rem/1 'DM Sans',sans-serif; }
        [data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"]::after { content:"‹"; }
        [data-testid="stExpandSidebarButton"]::after { content:"›"; }
        [data-testid="stSidebarUserContent"] > div { display:flex; flex-direction:column; height:calc(100vh - 5rem); min-height:0; }
        [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] { display:flex; flex:1; flex-direction:column; min-height:0; }
        [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:last-child { margin-top:auto; }
        h1,h2,h3 { color:var(--text)!important; font-family:'Space Grotesk',sans-serif; }
        .brand { padding:.9rem .4rem 1.5rem; border-bottom:1px solid #263d5a; margin-bottom:1.2rem; }
        .brand-icon { font-size:2.1rem; }.brand-kicker { color:var(--blue); font-size:.7rem; font-weight:700; letter-spacing:.2em; margin-top:.4rem; }.brand-name { color:var(--text); font:700 1.12rem/1.2 'Space Grotesk',sans-serif; letter-spacing:.05em; }.brand-sub { color:var(--muted); font-size:.77rem; margin-top:.55rem; }
        .eyebrow,.section-label { color:var(--blue); font-size:.7rem; font-weight:700; letter-spacing:.17em; margin-top:.8rem; }.page-title { font-size:clamp(2rem,4vw,3.45rem); margin:.2rem 0 .3rem; }.page-subtitle { color:var(--muted); font-size:1rem; margin-bottom:1.5rem; }
        .hero { background:linear-gradient(135deg,rgba(21,49,78,.97),rgba(18,27,53,.94)); border:1px solid #3474a0; border-radius:24px; padding:clamp(1.5rem,4vw,2.7rem); box-shadow:0 22px 55px rgba(0,0,0,.28),0 0 32px rgba(74,180,229,.08); }.hero-icon { font-size:2.3rem; }.hero-kicker { color:#7edcff; font-weight:700; letter-spacing:.14em; font-size:.75rem; }.hero-title { color:#fff; font:700 clamp(2rem,4vw,3.15rem)/1.08 'Space Grotesk',sans-serif; margin:.6rem 0 .8rem; }.hero-copy { color:#d2dcf1; line-height:1.7; max-width:760px; }
        .card,.kpi,.about-card,.upload-card,.flow-card,.quick-action { background:rgba(17,29,49,.86); border:1px solid var(--line); border-radius:16px; padding:1.2rem; height:100%; }.card { margin-top:1rem; min-height:115px; }.card-title,.about-heading { color:#fff; font-weight:700; }.card-copy,.about-body { color:var(--muted); font-size:.88rem; line-height:1.55; margin-top:.4rem; }.kpi { text-align:center; min-height:100px; box-shadow:0 10px 30px rgba(0,0,0,.12); }.kpi-value { color:#fff; font:700 1.75rem 'Space Grotesk',sans-serif; }.kpi-label { color:var(--muted); font-size:.75rem; margin-top:.25rem; }.kpi-positive { border-color:#397c61; }.kpi-neutral { border-color:#8b7131; }.kpi-negative { border-color:#85464f; }.quick-action { min-height:130px; border-color:#2c4c6d; box-shadow:0 12px 32px rgba(0,0,0,.16); transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease; }.quick-action:hover { transform:translateY(-2px); border-color:#58bfe9; box-shadow:0 14px 34px rgba(23,144,197,.12); }.quick-action-icon { font-size:1.5rem; }.quick-action-title { color:#f4f7fb; font-weight:700; margin-top:.35rem; }.quick-action-copy { color:var(--muted); font-size:.82rem; line-height:1.5; margin-top:.25rem; }.dataset-overview { display:flex; justify-content:space-between; align-items:center; gap:.75rem; flex-wrap:wrap; padding:1rem 1.2rem; margin-top:.5rem; background:rgba(17,29,49,.7); border:1px solid var(--line); border-radius:14px; color:var(--muted); font-size:.84rem; }.dataset-overview span:first-child { color:var(--blue); font-size:.7rem; font-weight:700; letter-spacing:.13em; }.dataset-overview strong { color:var(--text); }
        .sentiment-card { border-radius:17px; padding:1.2rem; border:1px solid; min-height:115px; }.positive { background:rgba(100,213,155,.1); border-color:#397c61; }.neutral { background:rgba(244,201,93,.1); border-color:#8b7131; }.negative { background:rgba(244,125,131,.1); border-color:#85464f; }.sentiment-icon { font-size:1.8rem; }.sentiment-name { color:#fff; font-weight:700; margin-top:.25rem; }.sentiment-copy { color:var(--muted); font-size:.8rem; margin-top:.3rem; }
        .pipeline { display:flex; justify-content:center; align-items:center; gap:.45rem; flex-wrap:wrap; margin:1rem 0 1.8rem; }.pipeline-step { background:var(--panel2); border:1px solid #355477; border-radius:11px; padding:.75rem .9rem; color:#dce8f8; font-size:.8rem; text-align:center; }.pipeline-arrow { color:var(--blue); font-size:1.2rem; }.feature { min-height:135px; }.feature-icon { font-size:1.5rem; }.feature-title { color:#fff; font-weight:700; margin-top:.6rem; }.feature-copy { color:var(--muted); font-size:.82rem; margin-top:.35rem; }.flow-card { text-align:center; }.flow-number { color:var(--purple); font:700 1.5rem 'Space Grotesk'; }
        .result { display:flex; gap:1rem; align-items:center; border-radius:19px; padding:1.35rem 1.5rem; margin:1.1rem 0; border:1px solid; }.result-positive { background:rgba(100,213,155,.12); border-color:#397c61; }.result-neutral { background:rgba(244,201,93,.12); border-color:#8b7131; }.result-negative { background:rgba(244,125,131,.12); border-color:#85464f; }.result-emoji { font-size:2.5rem; }.result-title { font:700 1.7rem 'Space Grotesk'; }.result-copy { color:var(--muted); margin-top:.25rem; }
        .upload-card { border-style:dashed; text-align:center; margin:.8rem 0 1.1rem; }.upload-card span { color:var(--muted); font-size:.85rem; }.stButton>button[kind="primary"] { background:linear-gradient(135deg,#68d69e,#6f9bff); color:#07101b; border:0; border-radius:10px; font-weight:700; min-height:2.7rem; }.stTextArea textarea,.stSelectbox div[data-baseweb="select"] { background:#101c30; color:var(--text); border-color:#345171; }.stDataFrame { border:1px solid #294362; border-radius:12px; overflow:hidden; }
        .login-page { min-height:86vh; display:flex; align-items:center; justify-content:center; padding:1.2rem 0; }.login-layout { width:min(1180px,100%); min-height:690px; display:grid; grid-template-columns:58% 42%; overflow:hidden; border-radius:28px; background:#f8faff; box-shadow:0 30px 90px rgba(0,0,0,.42); }        .login-visual { position:relative; overflow:hidden; padding:3.3rem; background:radial-gradient(circle at 18% 15%,rgba(255,255,255,.28),transparent 24%),linear-gradient(145deg,#38218b 0%,#5632c5 38%,#b348bf 72%,#f07caa 100%); }.login-visual:after { content:""; position:absolute; inset:18% -12% -22% 32%; border-radius:50%; background:linear-gradient(135deg,rgba(105,220,255,.28),rgba(255,255,255,.03)); transform:rotate(-16deg); }.visual-kicker { color:#d9d6ff; font-size:.76rem; font-weight:700; letter-spacing:.2em; position:relative; z-index:2; }.visual-title { color:#fff; font:700 clamp(2rem,4vw,3.8rem)/1.05 'Space Grotesk',sans-serif; max-width:440px; margin:1rem 0; position:relative; z-index:2; }.visual-copy { color:#eeeaff; max-width:440px; line-height:1.65; position:relative; z-index:2; }.visual-orbit { position:absolute; width:360px; height:360px; right:8%; bottom:7%; border:1px solid rgba(255,255,255,.24); border-radius:50%; box-shadow:0 0 0 34px rgba(255,255,255,.06),0 0 0 70px rgba(255,255,255,.04); }.visual-orbit:before,.visual-orbit:after { position:absolute; display:flex; align-items:center; justify-content:center; width:66px; height:66px; border-radius:22px; font-size:2rem; background:rgba(255,255,255,.2); box-shadow:0 13px 30px rgba(28,13,92,.25); }.visual-orbit:before { content:"😊"; top:8%; left:12%; }.visual-orbit:after { content:"📈"; right:6%; bottom:13%; }.visual-chart { position:absolute; z-index:2; right:16%; bottom:22%; width:235px; height:145px; padding:20px; border:1px solid rgba(255,255,255,.28); border-radius:18px; background:rgba(35,19,108,.38); backdrop-filter:blur(8px); }.chart-line { height:100%; display:flex; align-items:end; gap:12px; }.chart-bar { flex:1; border-radius:8px 8px 2px 2px; background:linear-gradient(180deg,#8be9ff,#6d75ff); }.chart-bar:nth-child(1){height:40%}.chart-bar:nth-child(2){height:68%}.chart-bar:nth-child(3){height:48%}.chart-bar:nth-child(4){height:86%}.chart-bar:nth-child(5){height:60%}.visual-insight { position:absolute; z-index:3; right:12%; bottom:8%; display:flex; gap:.45rem; }.visual-sentiment { display:flex; align-items:center; gap:.3rem; padding:.45rem .6rem; border:1px solid rgba(255,255,255,.25); border-radius:12px; background:rgba(255,255,255,.16); color:#fff; font-size:.72rem; backdrop-filter:blur(8px); }.visual-sentiment strong { font-size:.9rem; }.login-panel { display:flex; align-items:center; justify-content:center; padding:3.2rem 3.5rem; background:#fbfcff; }.login-card { width:100%; max-width:390px; text-align:left; }.login-icon { color:#714ce8; font-size:2.4rem; }.login-kicker { color:#6947d6; font-size:.72rem; font-weight:700; letter-spacing:.18em; margin-top:.75rem; }.login-title { color:#17152a; font:700 clamp(1.65rem,3vw,2.35rem)/1.08 'Space Grotesk',sans-serif; margin:.4rem 0 .65rem; }.login-sub { color:#74758a; line-height:1.5; margin-bottom:1.7rem; }.login-footer { color:#85869a; font-size:.78rem; margin-top:1.3rem; text-align:center; }.login-panel label { color:#38384b !important; font-weight:600; }.login-panel input { background:#f5f6fb !important; border:1px solid #e3e4ee !important; color:#24243a !important; border-radius:11px !important; }.login-panel .stButton>button { background:linear-gradient(100deg,#6a43dc,#4a8af3); color:white; border:0; border-radius:11px; min-height:2.9rem; font-weight:700; box-shadow:0 10px 20px rgba(87,76,205,.25); }.login-options { display:flex; justify-content:space-between; align-items:center; color:#77788c; font-size:.82rem; margin:.35rem 0 1rem; }.login-options [data-testid="stCheckbox"] { margin-top:-.4rem; }.login-options [data-testid="stCheckbox"] label { color:#77788c !important; font-weight:400; font-size:.82rem; }.login-forgot { color:#6947d6; font-weight:600; white-space:nowrap; }.login-divider { display:flex; align-items:center; gap:.8rem; color:#a0a1b0; font-size:.76rem; margin:1.35rem 0; }.login-divider:before,.login-divider:after { content:""; height:1px; background:#e1e2ec; flex:1; }.login-promise { text-align:center; color:#5d5e73; font-size:.85rem; font-weight:600; } 
        @media (max-width: 800px) { .login-layout { grid-template-columns:1fr; min-height:auto; }.login-visual { min-height:270px; padding:2rem; }.visual-orbit,.visual-chart { transform:scale(.62); transform-origin:bottom right; }.login-panel { padding:2.5rem 1.5rem; }.login-page { align-items:flex-start; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_login() -> None:
    st.markdown(
        '<div class="login-page"><div class="login-layout">'
        '<div class="login-visual"><div class="visual-kicker">🤖 REAL-TIME SENTIMENT ANALYSIS</div>'
        '<div class="visual-title">Understand the voice behind every sentence.</div>'
        '<div class="visual-copy">Turn reviews, feedback, comments, and opinions into clear sentiment intelligence.</div>'
        '<div class="visual-orbit"></div><div class="visual-chart"><div class="chart-line">'
        '<div class="chart-bar"></div><div class="chart-bar"></div><div class="chart-bar"></div>'
        '<div class="chart-bar"></div><div class="chart-bar"></div></div></div>'
        '<div class="visual-insight"><div class="visual-sentiment"><strong>😊</strong> Positive</div>'
        '<div class="visual-sentiment"><strong>😐</strong> Neutral</div><div class="visual-sentiment"><strong>☹</strong> Negative</div></div></div>'
        '<div class="login-panel"><div class="login-card"><div class="login-icon">🤖</div>'
        f'<div class="login-kicker">REAL-TIME</div><div class="login-title">SENTIMENT ANALYSIS<br>WITH WEB SCRAPING</div>'
        '<div class="login-sub">Intelligent Text &amp; Review Analysis Platform</div>',
        unsafe_allow_html=True,
    )
    with st.form("login_form"):
        username = st.text_input("👤 Username", placeholder="Enter username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter password")
        option_left, option_right = st.columns([1.1, 0.9])
        with option_left:
            st.checkbox("Remember me", key="remember_me")
        with option_right:
            st.markdown('<div class="login-forgot">Forgot password?</div>', unsafe_allow_html=True)
        submitted = st.form_submit_button("🔐 LOGIN", type="primary", use_container_width=True)
    st.markdown('<div class="login-divider">OR</div><div class="login-promise">Unlock insights from every piece of feedback</div></div></div></div></div>', unsafe_allow_html=True)
    if submitted:
        configured_username = os.environ.get("SENTIMENT_APP_USERNAME", "admin")
        configured_password = os.environ.get("SENTIMENT_APP_PASSWORD", "sentiment123")
        if username == configured_username and password == configured_password:
            st.session_state["authenticated"] = True
            st.query_params.clear()
            st.rerun()
        st.error("Invalid username or password")
    st.markdown('<div class="login-footer">Sentiment Intelligence Platform</div>', unsafe_allow_html=True)


def dashboard_counts() -> pd.Series | None:
    result = st.session_state.get("dataset_predictions")
    if not isinstance(result, pd.Series):
        return None
    labels = result.astype("string").str.strip().str.lower()
    labels = labels[labels.isin(LABELS)]
    if labels.empty:
        return None
    return labels.value_counts()


def render_dashboard() -> None:
    counts = dashboard_counts()
    source = st.session_state.get("dataset_source")
    st.markdown(
        f'<div class="hero"><div class="hero-kicker">{PROJECT_TITLE}</div>'
        f'<div class="hero-title">{PROJECT_TITLE}</div>'
        '<div class="hero-copy"><strong>Understand opinions instantly with NLP and Machine Learning</strong><br>'
        'Analyze individual reviews or complete datasets and visualize sentiment patterns through an interactive dashboard.</div></div>',
        unsafe_allow_html=True,
    )
    kpis = [
        ("📄", "Total Reviews", "—" if counts is None else f"{int(counts.sum()):,}", "kpi"),
        ("🟢", "Positive", "—" if counts is None else f"{int(counts.get('positive', 0)):,}", "kpi-positive"),
        ("🟡", "Neutral", "—" if counts is None else f"{int(counts.get('neutral', 0)):,}", "kpi-neutral"),
        ("🔴", "Negative", "—" if counts is None else f"{int(counts.get('negative', 0)):,}", "kpi-negative"),
    ]
    columns = st.columns(4)
    for column, (icon, label, value, css_class) in zip(columns, kpis):
        column.markdown(
            f'<div class="kpi {css_class}"><div class="sentiment-icon">{icon}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="section-label">SENTIMENT DISTRIBUTION</div>', unsafe_allow_html=True)
    if counts is None:
        st.info("Upload a dataset to view live statistics")
    else:
        chart_frame = pd.DataFrame(
            {
                "Sentiment": ["Positive", "Neutral", "Negative"],
                "Count": [
                    int(counts.get("positive", 0)),
                    int(counts.get("neutral", 0)),
                    int(counts.get("negative", 0)),
                ],
            }
        )
        left, right = st.columns(2)
        with left:
            fig = px.bar(
                chart_frame,
                x="Sentiment",
                y="Count",
                color="Sentiment",
                color_discrete_map={
                    "Positive": "#64d59b",
                    "Neutral": "#f4c95d",
                    "Negative": "#f47d83",
                },
                template="plotly_dark",
                title="Review Count",
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(l=10, r=10, t=45, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#dce8f8",
            )
            st.plotly_chart(fig, width="stretch")
        with right:
            fig = px.pie(
                chart_frame,
                names="Sentiment",
                values="Count",
                hole=0.62,
                color="Sentiment",
                color_discrete_map={
                    "Positive": "#64d59b",
                    "Neutral": "#f4c95d",
                    "Negative": "#f47d83",
                },
                template="plotly_dark",
                title="Sentiment Share",
            )
            fig.update_layout(
                showlegend=True,
                margin=dict(l=10, r=10, t=45, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#dce8f8",
            )
            st.plotly_chart(fig, width="stretch")
        st.markdown(
            f'<div class="dataset-overview"><span>DATASET OVERVIEW</span>'
            f'<strong>{int(counts.sum()):,} reviews analyzed</strong>'
            f'<span>{source if isinstance(source, str) and source else "Session dataset"}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">QUICK ACTIONS</div>', unsafe_allow_html=True)
    actions = [
        ("✍️", "Analyze Single Review", "Classify an individual review instantly.", "✍️ Single Review"),
        ("📊", "Analyze Dataset", "Explore sentiment across a review dataset.", "📊 Dataset Analysis"),
        ("🕘", "View History", "Review analyses from this session.", "🕘 History"),
    ]
    columns = st.columns(3)
    for column, (icon, title, description, page) in zip(columns, actions):
        with column:
            st.markdown(
                f'<div class="quick-action"><div class="quick-action-icon">{icon}</div>'
                f'<div class="quick-action-title">{title}</div>'
                f'<div class="quick-action-copy">{description}</div></div>',
                unsafe_allow_html=True,
            )
            if st.button(title, key=f"home_action_{page}", width="stretch"):
                set_current_page(page)
                st.rerun()

    st.markdown('<div class="section-label">HOW IT WORKS</div>', unsafe_allow_html=True)
    steps = ["Review Collection", "Data Processing", "Sentiment Prediction", "Visual Insights"]
    pipeline = '<div class="pipeline">'
    for index, step in enumerate(steps):
        pipeline += f'<div class="pipeline-step">{index + 1:02d}<br>{step}</div>'
        if index < len(steps) - 1:
            pipeline += '<div class="pipeline-arrow">→</div>'
    st.markdown(pipeline + "</div>", unsafe_allow_html=True)


def render_single_review_redesign(model) -> None:
    st.markdown('<div class="eyebrow">✍️ TEXT INTELLIGENCE</div><h1 class="page-title">Analyze Single Text</h1><p class="page-subtitle">Enter any review, comment, feedback, or text to analyze its sentiment.</p>', unsafe_allow_html=True)
    with st.form("single_review_form_redesign"):
        review = st.text_area("Enter your text here...", height=190, placeholder="Example: The product quality is excellent and I really love it.", label_visibility="visible")
        submitted = st.form_submit_button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    if not submitted:
        return
    text = review.strip()
    normalized_text = normalize_review_phrases(text)
    if not has_meaningful_review_text(normalized_text, model):
        st.warning(
            "⚠️ Please enter meaningful review text using words recognized by the sentiment model."
        )
        return
    predicted, probabilities = predict_one(model, normalized_text)
    st.session_state.setdefault("history", []).append({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "review": text, "predicted_sentiment": display_sentiment(predicted), "positive_probability": probabilities["positive"] if probabilities is not None else None, "neutral_probability": probabilities["neutral"] if probabilities is not None else None, "negative_probability": probabilities["negative"] if probabilities is not None else None})
    presentation = {"positive": ("🟢", "POSITIVE", "Positive sentiment detected", "result-positive"), "neutral": ("🟡", "NEUTRAL", "Neutral sentiment detected", "result-neutral"), "negative": ("🔴", "NEGATIVE", "Negative sentiment detected", "result-negative")}
    emoji, title, message, css_class = presentation[predicted]
    st.markdown(f'<div class="result {css_class}"><div class="result-emoji">{emoji}</div><div><div class="result-title">{title}</div><div class="result-copy">{message}</div></div></div>', unsafe_allow_html=True)
    if probabilities is not None:
        cols = st.columns(3)
        cols[0].metric("Positive probability", f"{probabilities['positive']:.2%}")
        cols[1].metric("Neutral probability", f"{probabilities['neutral']:.2%}")
        cols[2].metric("Negative probability", f"{probabilities['negative']:.2%}")


def render_dataset_redesign(model) -> None:
    st.markdown('<div class="eyebrow">📊 TEXT INTELLIGENCE</div><h1 class="page-title">Dataset Analysis</h1><p class="page-subtitle">Analyze reviews, feedback, comments, survey responses, and other text at scale.</p>', unsafe_allow_html=True)
    mode = st.radio("Choose a dataset", ["Upload CSV", "Women's Clothing dataset"], horizontal=True, key="dataset_analysis_mode_redesign")
    if mode == "Women's Clothing dataset":
        st.session_state.pop("dataset_predictions", None)
        st.session_state.pop("dataset_source", None)
        if not CLOTHING_PATH.is_file():
            st.error(f"Dataset not found: {CLOTHING_PATH}")
            return
        frame = pd.read_csv(CLOTHING_PATH, usecols=["Review Text", "Rating"])
        actual, invalid_count = map_ground_truth(frame["Rating"], "Rating")
        valid = frame["Review Text"].fillna("").astype(str).str.strip().ne("") & actual.notna()
        texts, actual = frame.loc[valid, "Review Text"].fillna("").astype(str).str.strip(), actual.loc[valid].astype(str)
        if invalid_count:
            st.warning(f"{invalid_count:,} rows have missing or invalid ratings.")
        if texts.empty:
            st.error("No valid reviews found.")
            return
        predictions = pd.Series(model.predict(texts.to_numpy()), index=texts.index)
        st.session_state["dataset_predictions"] = predictions.copy()
        st.session_state["dataset_source"] = CLOTHING_PATH.name
        st.caption(f"Rating mapping: 1–2 Negative • 3 Neutral • 4–5 Positive • {len(texts):,} reviews analyzed")
        render_dataset_summary_redesign(predictions)
        render_evaluation_metrics(actual, predictions)
        st.dataframe(pd.DataFrame({"Review": texts, "Actual": actual.map(display_sentiment), "Predicted Sentiment": predictions.map(display_sentiment)}), hide_index=True, width="stretch")
        return
    st.markdown('<div class="upload-card">📁 <strong>Upload Review Dataset</strong><br><span>Choose a CSV containing a text or review column.</span></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose CSV", type=["csv"], key="dataset_csv_upload_redesign", label_visibility="collapsed")
    if uploaded is None:
        st.info("Upload a CSV to begin analysis.")
        return
    st.session_state.pop("dataset_predictions", None)
    st.session_state.pop("dataset_source", None)
    try:
        frame = pd.read_csv(BytesIO(uploaded.getvalue()))
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
        st.error(f"Could not read the uploaded CSV: {error}")
        return
    text_options = text_column_options(frame)
    if not text_options:
        st.error("No text column detected.")
        return
    text_column = st.selectbox("Review text column", text_options, key="dataset_text_column_redesign")
    label_options = ["No ground-truth labels"] + [column for column in frame.columns if column != text_column]
    label_column = st.selectbox("Actual sentiment/rating column", label_options, index=0, key="dataset_label_column_redesign")
    texts = frame[text_column].fillna("").astype(str).str.strip()
    nonempty = texts.ne("")
    predictions = pd.Series(index=frame.index, dtype="string")
    if nonempty.any():
        predictions.loc[nonempty] = model.predict(texts.loc[nonempty].to_numpy())
        st.session_state["dataset_predictions"] = predictions.loc[nonempty].copy()
        st.session_state["dataset_source"] = uploaded.name
        render_dataset_summary_redesign(predictions.loc[nonempty])
    result = pd.DataFrame({"Review": texts, "Predicted Sentiment": predictions.map(display_sentiment)})
    if label_column != "No ground-truth labels":
        actual, invalid_count = map_ground_truth(frame[label_column], str(label_column))
        valid_actual = nonempty & actual.notna()
        result["Actual"] = actual.map(display_sentiment)
        if invalid_count:
            st.warning(f"{invalid_count:,} labels could not be mapped and are excluded from metrics.")
        if valid_actual.any():
            render_evaluation_metrics(actual.loc[valid_actual], predictions.loc[valid_actual])
    else:
        st.info("No ground-truth labels selected; evaluation metrics are not calculated.")
    st.dataframe(result, hide_index=True, width="stretch")


def render_dataset_summary_redesign(predictions: pd.Series) -> None:
    counts = predictions.astype("string").value_counts()
    cards = [("📄", "Total Texts", int(counts.sum()), "kpi"), ("🟢", "Positive", int(counts.get("positive", 0)), "kpi-positive"), ("🟡", "Neutral", int(counts.get("neutral", 0)), "kpi-neutral"), ("🔴", "Negative", int(counts.get("negative", 0)), "kpi-negative")]
    cols = st.columns(4)
    for col, (icon, label, value, css_class) in zip(cols, cards):
        col.markdown(f'<div class="kpi {css_class}"><div class="sentiment-icon">{icon}</div><div class="kpi-value">{value:,}</div><div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)
    chart_data = pd.DataFrame({"Sentiment": ["Positive", "Neutral", "Negative"], "Count": [counts.get("positive", 0), counts.get("neutral", 0), counts.get("negative", 0)]})
    fig = px.bar(chart_data, x="Sentiment", y="Count", color="Sentiment", color_discrete_map={"Positive": "#64d59b", "Neutral": "#f4c95d", "Negative": "#f47d83"}, template="plotly_dark")
    fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_history_redesign() -> None:
    st.markdown('<div class="eyebrow">🕘 ACTIVITY LOG</div><h1 class="page-title">Recent Analyses</h1><p class="page-subtitle">Review the text analyzed during this session.</p>', unsafe_allow_html=True)
    history = st.session_state.get("history", [])
    if not history:
        st.info("No analyses yet. Analyze a single text to create history.")
        return
    frame = pd.DataFrame(history).rename(columns={"review": "Review", "predicted_sentiment": "Sentiment", "timestamp": "Time"})
    frame["Sentiment"] = frame["Sentiment"].map({"Positive": "🟢 Positive", "Neutral": "🟡 Neutral", "Negative": "🔴 Negative"})
    st.dataframe(frame[["Review", "Sentiment", "Time"]], hide_index=True, width="stretch")
    st.download_button("Download history CSV", frame.to_csv(index=False).encode("utf-8"), file_name="sentiment_history.csv", mime="text/csv")


def render_about_redesign() -> None:
    st.markdown(f'<div class="eyebrow">ℹ️ PROJECT OVERVIEW</div><h1 class="page-title">About {PROJECT_TITLE}</h1><p class="page-subtitle">A general-purpose platform for understanding written feedback.</p>', unsafe_allow_html=True)
    sections = [
        ("🎯 Objective", "Analyze customer feedback, reviews, comments, survey responses, and other text to make opinions easier to understand."),
        ("🔍 Problem Statement", "Manually reviewing large volumes of text is slow and makes it difficult to identify broad sentiment patterns."),
        ("💡 Solution", "Automated sentiment classification turns unstructured text into clear Positive, Neutral, and Negative insights."),
        ("🧠 Sentiment Classification", "🟢 Positive   •   🟡 Neutral   •   🔴 Negative"),
        ("🛠️ Technologies", "Python • Pandas • NLTK • Scikit-learn • TF-IDF • Logistic Regression • Streamlit • Plotly"),
        ("🌐 Applications", "Customer feedback • Product reviews • Survey responses • Comments • Business feedback • General text analysis"),
        ("🚀 Future Scope", "Real-time data collection • Advanced NLP models • Aspect-based analysis • Multi-language sentiment • Live analytics"),
        (
            "🧠 Machine Learning Algorithms",
            "<strong>Logistic Regression</strong><br>Primary classifier used with TF-IDF features for three-class sentiment classification."
            "<br><br><strong>Multinomial Naive Bayes</strong><br>Comparison model used during development for text classification."
            "<br><br><strong>Linear SVM</strong><br>Comparison model used during development for high-dimensional TF-IDF text features."
            "<br><br><strong>Clarification:</strong> TF-IDF is a feature extraction technique, not a Machine Learning algorithm."
            "<br><br>Sentiment classes: Positive • Neutral • Negative",
        ),
    ]
    for index in range(0, len(sections), 2):
        cols = st.columns(2)
        for col, (title, body) in zip(cols, sections[index:index + 2]):
            col.markdown(f'<div class="about-card"><div class="about-heading">{title}</div><div class="about-body">{body}</div></div>', unsafe_allow_html=True)


def new_main() -> None:
    st.set_page_config(page_title=PROJECT_TITLE, page_icon="🤖", layout="wide")
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if not st.session_state["authenticated"]:
        show_login_page()
        st.stop()
    _new_theme()
    try:
        model, metadata = load_artifacts()
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        st.error(f"Could not load the saved sentiment system: {error}")
        st.stop()
    st.sidebar.markdown(
        f'<div class="brand"><div class="brand-icon">🤖</div><div class="brand-kicker">{PROJECT_TITLE}</div><div class="brand-sub">Intelligent Text &amp; Review Analysis</div></div>',
        unsafe_allow_html=True,
    )
    current_page = resolve_current_page()
    selected_page = st.sidebar.radio(
        "Navigation",
        PAGE_OPTIONS,
        index=PAGE_OPTIONS.index(current_page),
        key="navigation_page",
        label_visibility="collapsed",
    )
    st.session_state["selected_page"] = selected_page
    st.query_params["page"] = selected_page
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state.pop("selected_page", None)
        st.query_params.clear()
        st.rerun()
    if selected_page == "🏠 Home":
        render_dashboard()
    elif selected_page == "✍️ Single Review":
        render_single_review_redesign(model)
    elif selected_page == "📊 Dataset Analysis":
        render_dataset_redesign(model)
    elif selected_page == "🕘 History":
        render_history_redesign()
    elif selected_page == "ℹ️ About":
        render_about_redesign()


def main() -> None:
    new_main()
    return
    st.set_page_config(page_title=PROJECT_TITLE, page_icon="🤖", layout="wide")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root { --bg: #0b1120; --panel: #121c2e; --panel-2: #17243a; --muted: #91a0b7; --text: #f4f7fb; --green: #64d59b; --yellow: #f4c95d; --red: #f47d83; --blue: #73a7ff; }
        .stApp { background: radial-gradient(circle at 80% 0%, #183354 0, transparent 35%), var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #111d31 0%, #0b1322 100%); border-right: 1px solid #233652; }
        [data-testid="stSidebar"] * { font-family: 'DM Sans', sans-serif; }
        [data-testid="stSidebar"] .stRadio label { padding: 0.55rem 0.65rem; border-radius: 10px; }
        [data-testid="stSidebar"] .stRadio label:hover { background: #1c2b43; }
        h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; color: var(--text) !important; }
        .brand { padding: 1.1rem 0.55rem 1.7rem; border-bottom: 1px solid #233652; margin-bottom: 1.35rem; }
        .brand-mark { font-size: 2.1rem; }.brand-kicker { color: var(--green); font-size: 0.72rem; font-weight: 700; letter-spacing: .18em; margin-top: .5rem; }
        .brand-title { color: var(--text); font: 700 1.15rem/1.2 'Space Grotesk', sans-serif; letter-spacing: .04em; }.brand-subtitle { color: var(--muted); font-size: .78rem; margin-top: .55rem; }
        .page-eyebrow, .section-label { color: var(--green); font-size: .72rem; font-weight: 700; letter-spacing: .16em; margin-top: .8rem; }
        .page-title { font-size: clamp(2rem, 4vw, 3.55rem); margin: .25rem 0 .35rem; }.page-subtitle { color: var(--muted); font-size: 1.05rem; margin-bottom: 1.8rem; }
        .hero { background: linear-gradient(135deg, rgba(35, 69, 102, .9), rgba(20, 35, 59, .92)); border: 1px solid #2b4c70; border-radius: 24px; padding: 2.6rem; box-shadow: 0 24px 60px rgba(0,0,0,.22); }
        .hero-kicker { color: var(--green); font-weight: 700; letter-spacing: .12em; font-size: .78rem; }.hero-title { color: white; font: 700 clamp(2rem, 4vw, 3.5rem)/1.08 'Space Grotesk', sans-serif; margin: .7rem 0 1rem; }.hero-copy { color: #c1ccdc; font-size: 1.05rem; max-width: 720px; line-height: 1.7; }
        .info-card, .feature-card, .flow-card, .about-card, .summary-card { background: rgba(18, 28, 46, .88); border: 1px solid #263d5a; border-radius: 16px; padding: 1.2rem; height: 100%; }.info-card { margin-top: 1rem; min-height: 125px; }.info-title, .feature-title, .flow-title, .about-heading { color: white; font-weight: 700; font-size: 1rem; }.info-copy, .feature-copy, .about-body { color: var(--muted); font-size: .9rem; line-height: 1.55; margin-top: .45rem; }
        .sentiment-card { border-radius: 18px; padding: 1.35rem; min-height: 120px; border: 1px solid; }.sentiment-positive { background: rgba(100,213,155,.1); border-color: rgba(100,213,155,.35); }.sentiment-neutral { background: rgba(244,201,93,.1); border-color: rgba(244,201,93,.35); }.sentiment-negative { background: rgba(244,125,131,.1); border-color: rgba(244,125,131,.35); }.sentiment-icon { font-size: 1.7rem; }.sentiment-name { font-weight: 700; margin-top: .35rem; }.sentiment-desc { color: var(--muted); font-size: .82rem; margin-top: .3rem; }
        .pipeline { display: flex; align-items: center; justify-content: center; gap: .45rem; flex-wrap: wrap; margin: 1rem 0 2rem; }.pipeline-step { background: #172943; border: 1px solid #2a4768; border-radius: 12px; padding: .85rem 1rem; color: #d8e3f1; font-size: .85rem; text-align: center; }.pipeline-arrow { color: var(--green); font-size: 1.2rem; }
        .feature-card { min-height: 145px; }.feature-icon { font-size: 1.65rem; margin-bottom: .65rem; }.flow-card { text-align: center; }.flow-number { color: var(--green); font: 700 1.5rem 'Space Grotesk', sans-serif; }
        .result-card { display: flex; align-items: center; gap: 1rem; border-radius: 18px; padding: 1.25rem 1.5rem; margin: 1.2rem 0; border: 1px solid; }.result-positive { background: rgba(100,213,155,.12); border-color: #3d986c; }.result-neutral { background: rgba(244,201,93,.12); border-color: #ae8d37; }.result-negative { background: rgba(244,125,131,.12); border-color: #a75059; }.result-icon { font-size: 2.25rem; }.result-title { font: 700 1.6rem 'Space Grotesk', sans-serif; }.result-message { color: var(--muted); margin-top: .25rem; }
        .upload-panel { background: #172943; border: 1px dashed #49749e; border-radius: 16px; padding: 1rem 1.2rem; margin: .7rem 0 1rem; color: #e4edf8; }.upload-panel span { color: var(--muted); font-size: .85rem; }.summary-card { text-align: center; min-height: 105px; }.summary-value { color: white; font: 700 1.65rem 'Space Grotesk', sans-serif; }.summary-label { color: var(--muted); font-size: .76rem; margin-top: .35rem; }.summary-positive { border-color: rgba(100,213,155,.45); }.summary-negative { border-color: rgba(244,125,131,.45); }
        .stButton > button[kind="primary"] { background: linear-gradient(135deg, #5acb95, #3997c9); border: 0; color: #07101b; font-weight: 700; border-radius: 10px; min-height: 2.7rem; }.stTextArea textarea, .stSelectbox div[data-baseweb="select"] { background: #111d31; color: var(--text); border-color: #2a4768; }.stDataFrame { border: 1px solid #263d5a; border-radius: 12px; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    try:
        model, metadata = load_artifacts()
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        st.error(f"Could not load the latest saved model and metadata: {error}")
        st.stop()

    st.sidebar.markdown(
        '<div class="brand"><div class="brand-mark">🤖</div><div class="brand-kicker">REAL-TIME</div>'
        '<div class="brand-title">SENTIMENT<br>ANALYSIS</div><div class="brand-subtitle">E-Commerce Review Intelligence</div></div>',
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Home", "✍️ Single Review", "📊 Dataset Analysis", "🕘 History", "ℹ️ About"],
        key="navigation",
    )
    if page == "🏠 Home":
        render_home()
    elif page == "✍️ Single Review":
        render_single_review(model)
    elif page == "📊 Dataset Analysis":
        render_dataset_analysis(model)
    elif page == "🕘 History":
        render_history()
    elif page == "ℹ️ About":
        render_about(metadata, model)


if __name__ == "__main__":
    main()
