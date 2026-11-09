
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("outputs/lapse_dataset.csv")

print("Dataset shape:", df.shape)
print("\nLapse distribution:")
print(df["lapsed"].value_counts())

# --------------------------------------------------
# 2. SELECT FEATURES AND TARGET
# --------------------------------------------------

# Use only the features specified in the assignment.
# Exclude customer_id and lapsed from the input features.

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

feature_columns = numeric_features + categorical_features

X = df[feature_columns]
y = df["lapsed"].astype(int)

print("\nFeatures used:", feature_columns)
print("Target: lapsed")

# --------------------------------------------------
# 3. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

# --------------------------------------------------
# 4. PREPROCESSING
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

print("\nPreprocessing pipeline created successfully!")

# --------------------------------------------------
# 5. LOGISTIC REGRESSION
# --------------------------------------------------

logistic_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

logistic_parameters = {
    "classifier__C": [0.01, 0.1, 1, 10, 100]
}

logistic_grid = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid=logistic_parameters,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

print("\nTraining Logistic Regression...")

logistic_grid.fit(X_train, y_train)

best_logistic = logistic_grid.best_estimator_
logistic_predictions = best_logistic.predict(X_test)

print("\n===== LOGISTIC REGRESSION RESULTS =====")
print("Best parameters:", logistic_grid.best_params_)
print("Best CV F1:", round(logistic_grid.best_score_, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, logistic_predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    logistic_predictions,
    zero_division=0
))

logistic_f1 = f1_score(
    y_test, logistic_predictions, pos_label=1
)

# --------------------------------------------------
# 6. RANDOM FOREST
# --------------------------------------------------

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

rf_parameters = {
    "classifier__max_depth": [3, 5, 8],
    "classifier__min_samples_leaf": [5, 10, 20]
}

rf_grid = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_parameters,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

print("\nTraining Random Forest...")

rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_
rf_predictions = best_rf.predict(X_test)

print("\n===== RANDOM FOREST RESULTS =====")
print("Best parameters:", rf_grid.best_params_)
print("Best CV F1:", round(rf_grid.best_score_, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

print("\nClassification Report:")
print(classification_report(
    y_test,
    rf_predictions,
    zero_division=0
))

rf_f1 = f1_score(
    y_test, rf_predictions, pos_label=1
)

# --------------------------------------------------
# 7. MODEL COMPARISON
# --------------------------------------------------

comparison = pd.DataFrame([
    {
        "model": "Logistic Regression",
        "accuracy": accuracy_score(y_test, logistic_predictions),
        "precision_lapsed": precision_score(
            y_test, logistic_predictions, pos_label=1,
            zero_division=0
        ),
        "recall_lapsed": recall_score(
            y_test, logistic_predictions, pos_label=1,
            zero_division=0
        ),
        "f1_lapsed": logistic_f1
    },
    {
        "model": "Random Forest",
        "accuracy": accuracy_score(y_test, rf_predictions),
        "precision_lapsed": precision_score(
            y_test, rf_predictions, pos_label=1,
            zero_division=0
        ),
        "recall_lapsed": recall_score(
            y_test, rf_predictions, pos_label=1,
            zero_division=0
        ),
        "f1_lapsed": rf_f1
    }
])

print("\n===== FINAL MODEL COMPARISON =====")
print(comparison.round(4).to_string(index=False))

# Save comparison results
comparison.to_csv(
    "outputs/lapse_model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")
print("Output: outputs/lapse_model_comparison.csv")



# --------------------------------------------------
# 8. RANDOM FOREST - TOP 5 FEATURE IMPORTANCE
# --------------------------------------------------

# Get feature names after preprocessing
feature_names = best_rf.named_steps["preprocessor"].get_feature_names_out()

# Get feature importance from trained Random Forest
importances = best_rf.named_steps["classifier"].feature_importances_

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\n===== TOP 5 RANDOM FOREST FEATURES =====")
print(feature_importance.head(5).to_string(index=False))

# Save feature importance
feature_importance.to_csv(
    "outputs/rf_feature_importance.csv",
    index=False
)


# --------------------------------------------------
# 9. LOGISTIC REGRESSION - COEFFICIENT ANALYSIS
# --------------------------------------------------

# Get feature names after preprocessing
logistic_feature_names = (
    best_logistic.named_steps["preprocessor"].get_feature_names_out()
)

# Get coefficients
coefficients = best_logistic.named_steps["classifier"].coef_[0]

coefficient_df = pd.DataFrame({
    "feature": logistic_feature_names,
    "coefficient": coefficients
})

# Sort by absolute coefficient value
coefficient_df["absolute_coefficient"] = (
    coefficient_df["coefficient"].abs()
)

coefficient_df = coefficient_df.sort_values(
    by="absolute_coefficient",
    ascending=False
)

print("\n===== TOP 3 LOGISTIC REGRESSION COEFFICIENTS =====")
print(
    coefficient_df[
        ["feature", "coefficient"]
    ].head(3).to_string(index=False)
)

# Save coefficients
coefficient_df.to_csv(
    "outputs/logistic_coefficients.csv",
    index=False
)

print("\nFeature importance and coefficient analysis saved!")
