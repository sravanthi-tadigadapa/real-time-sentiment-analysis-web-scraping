# Amazon review sentiment model

Train and evaluate a three-class sentiment model using the cleaned Amazon reviews and the women's clothing reviews:

```powershell
python train_sentiment_model.py
```

The script checks the Amazon sentiment labels against the review scores, reserves a stratified Amazon holdout, and creates a reproducibly sampled balanced training set. It compares Logistic Regression, Multinomial Naive Bayes, and Linear SVM using validation macro-F1, then refits the highest-scoring model on the full balanced sample. The final model is evaluated on the untouched Amazon holdout and on non-empty women's clothing reviews; clothing labels are derived from ratings (1-2 negative, 3 neutral, 4-5 positive).

The fitted TF-IDF and classifier pipeline is saved as `sentiment_model.pkl`. Override the input or output paths with `--amazon-csv`, `--clothing-csv`, or `--output`; use `--samples-per-class` to change the maximum number of Amazon training reviews per sentiment.

Load the saved pipeline with `joblib.load("sentiment_model.pkl")` and call `predict` with an iterable of review texts. Predictions are `negative`, `neutral`, or `positive`.

## Streamlit application

Launch the review analyzer from the project directory:

```powershell
python -m streamlit run app.py
```

The app loads only `model/sentiment_model.pkl` and verifies it against `model/model_metadata.json`; it does not fall back to the older root-level model. It provides Home, Single Review, Dataset Analysis, Model Performance, History, and About pages. Dataset Analysis accepts CSV uploads and can evaluate the bundled `womens_clothing_reviews.csv` using the verified rating mapping. Uploaded or bundled data is used only for predictions and evaluation, never training. Single-review history is held in the current Streamlit session.

## Model validation and improvement study

Run the Amazon-only feature and classifier comparison with:

```powershell
python improve_sentiment_model.py
```

This longer-running study uses five-fold stratified cross-validation on a reproducible Amazon training sample to compare word bigrams, word trigrams, character n-grams, combined word/character features, Logistic Regression, LinearSVC, and Multinomial Naive Bayes. It retains all Neutral examples in the final fit sample and does not use the women's clothing reviews for model selection. That dataset is evaluated only after selecting the Amazon CV winner.

The script compares the CV winner against the existing word-bigram Logistic Regression configuration. It saves `model/sentiment_model.pkl` only when the winner has higher five-fold CV macro-F1 than that baseline; the original root-level model is not overwritten. `model/model_metadata.json` records the comparison, CV results, holdout metrics, and per-class metrics. `model/neutral_training_audit.csv` contains 100 real Neutral training examples, while `model/rating_text_disagreement_audit.csv` contains held-out reviews flagged for manual rating/text review. A disagreement with a model prediction is an audit candidate, not proof that a dataset label is wrong.
