
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("outputs/lapse_dataset.csv")

numeric_features = [
    "frequency",
    "monetary",
    "recency",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery_time",
    "age",
    "tenure_days"
]

categorical_features = [
    "favourite_category",
    "city",
    "gender"
]

X = df[numeric_features + categorical_features]
y = df["lapsed"].astype(int)

# --------------------------------------------------
# 2. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Original training distribution:")
print(y_train.value_counts())

# --------------------------------------------------
# 3. PREPROCESSING
# --------------------------------------------------

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# --------------------------------------------------
# 4. BASELINE LOGISTIC REGRESSION
# --------------------------------------------------

baseline_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        C=10,
        max_iter=1000
    ))
])

print("\nTraining baseline Logistic Regression...")

baseline_model.fit(X_train, y_train)
baseline_predictions = baseline_model.predict(X_test)

# --------------------------------------------------
# 5. OVERSAMPLED LOGISTIC REGRESSION
# --------------------------------------------------

oversampled_model = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("oversampler", RandomOverSampler(random_state=42)),
    ("classifier", LogisticRegression(
        C=10,
        max_iter=1000
    ))
])

print("\nTraining Logistic Regression with oversampling...")

oversampled_model.fit(X_train, y_train)

oversampled_predictions = oversampled_model.predict(X_test)

# --------------------------------------------------
# 6. DISPLAY BALANCED TRAINING DISTRIBUTION
# --------------------------------------------------

balanced_y = oversampled_model.named_steps[
    "oversampler"
].fit_resample(
    oversampled_model.named_steps[
        "preprocessor"
    ].transform(X_train),
    y_train
)[1]

print("\nBalanced training distribution:")
print(pd.Series(balanced_y).value_counts())

# --------------------------------------------------
# 7. EVALUATE BOTH MODELS
# --------------------------------------------------

def evaluate_model(name, actual, predicted):

    print(f"\n===== {name} =====")

    print("\nConfusion Matrix:")
    print(confusion_matrix(actual, predicted))

    print("\nClassification Report:")
    print(classification_report(
        actual,
        predicted,
        zero_division=0
    ))

    return {
        "model": name,
        "accuracy": accuracy_score(actual, predicted),
        "precision_lapsed": precision_score(
            actual, predicted, pos_label=1, zero_division=0
        ),
        "recall_lapsed": recall_score(
            actual, predicted, pos_label=1, zero_division=0
        ),
        "f1_lapsed": f1_score(
            actual, predicted, pos_label=1, zero_division=0
        )
    }


baseline_results = evaluate_model(
    "Baseline Logistic Regression",
    y_test,
    baseline_predictions
)

oversampled_results = evaluate_model(
    "Oversampled Logistic Regression",
    y_test,
    oversampled_predictions
)

# --------------------------------------------------
# 8. FINAL COMPARISON
# --------------------------------------------------

comparison = pd.DataFrame([
    baseline_results,
    oversampled_results
])

print("\n===== OVERSAMPLING COMPARISON =====")
print(comparison.round(4).to_string(index=False))

comparison.to_csv(
    "outputs/lapse_oversampling_comparison.csv",
    index=False
)

print("\nComparison saved successfully!")

# --------------------------------------------------
# 9. SAVE SELECTED MODEL
# --------------------------------------------------

import joblib
import numpy as np

# Save the oversampled Logistic Regression pipeline
joblib.dump(
    oversampled_model,
    "outputs/lapse_model.joblib"
)

print("\nSelected model saved successfully!")


# --------------------------------------------------
# 10. RELOAD MODEL AND VERIFY PREDICTIONS
# --------------------------------------------------

loaded_model = joblib.load(
    "outputs/lapse_model.joblib"
)

# Generate predictions using the original model
original_predictions = oversampled_model.predict(X_test)

# Generate predictions using the reloaded model
reloaded_predictions = loaded_model.predict(X_test)

# Verify both predictions are identical
predictions_match = np.array_equal(
    original_predictions,
    reloaded_predictions
)

print("\n===== MODEL RELOAD VALIDATION =====")
print("Original predictions:", len(original_predictions))
print("Reloaded predictions:", len(reloaded_predictions))
print("Predictions identical:", predictions_match)

assert predictions_match, "Model predictions do not match!"

print("\nALL MODEL SAVING CHECKPOINTS PASSED")