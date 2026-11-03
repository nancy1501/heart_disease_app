"""
Heart Disease Prediction - Model Training & Evaluation
========================================================
Models:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. Random Forest Classifier

Visualisations: Accuracy, F1-Score, Recall, Precision + Confusion Matrices
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ── 1. Load dataset ───────────────────────────────────────────────────────────
df = pd.read_csv("heart.csv")
print("=" * 65)
print("  HEART DISEASE PREDICTION  -  MODEL TRAINING & EVALUATION")
print("=" * 65)
print(f"\n  Dataset shape : {df.shape}")
print(f"  Target dist.  : {dict(df['HeartDisease'].value_counts())}\n")

# ── 2. Data cleaning (same as notebook) ────────────────────────────────────────
for col in ["Cholesterol", "RestingBP"]:
    mean_val = df.loc[df[col] != 0, col].mean()
    df[col] = df[col].replace(0, mean_val)
    print(f"  Replaced zeros in '{col}' with mean = {mean_val:.2f}")

# ── 3. Encoding (same as notebook) ────────────────────────────────────────────
df_encode = pd.get_dummies(df, drop_first=True)

# ── 4. Scaling numerical features ─────────────────────────────────────────────
numerical_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
scaler = StandardScaler()
df_encode[numerical_cols] = scaler.fit_transform(df_encode[numerical_cols])

print(f"\n  Encoded dataset : {df_encode.shape[0]} rows x {df_encode.shape[1]} cols")

# ── 5. Train / Test split ────────────────────────────────────────────────────
X = df_encode.drop("HeartDisease", axis=1)
y = df_encode["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Train size      : {X_train.shape[0]}")
print(f"  Test  size      : {X_test.shape[0]}")

# ── 6. Define models ─────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# ── 7. Train, predict & evaluate ──────────────────────────────────────────────
results = {}

for name, model in models.items():
    print("\n" + "-" * 65)
    print(f"  MODEL: {name}")
    print("-" * 65)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)

    results[name] = {
        "accuracy": acc,
        "f1": f1,
        "recall": rec,
        "precision": prec,
    }

    print(f"\n  Accuracy   : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1-Score   : {f1:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  Precision  : {prec:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["No Disease", "Disease"]))
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:\n{cm}\n")

# ── 8. Feature importance (Random Forest) ────────────────────────────────────
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

print("=" * 65)
print("  RANDOM FOREST  -  TOP FEATURE IMPORTANCES")
print("=" * 65)
for feat, imp in feat_imp.items():
    print(f"    {feat:25s}  {imp:.4f}")

# ── 9. Model comparison table ────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  MODEL COMPARISON SUMMARY")
print("=" * 65)
print(f"  {'Model':25s}  {'Accuracy':>8s}  {'F1':>8s}  {'Recall':>8s}  {'Precision':>9s}")
print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*9}")
for name in results:
    r = results[name]
    print(f"  {name:25s}  {r['accuracy']*100:7.2f}%  {r['f1']*100:7.2f}%  "
          f"{r['recall']*100:7.2f}%  {r['precision']*100:8.2f}%")

best_model = max(results, key=lambda k: results[k]["accuracy"])
print(f"\n  * Best model (accuracy): {best_model} "
      f"({results[best_model]['accuracy']*100:.2f}%)")

# ── 10. Visualisations ───────────────────────────────────────────────────────

model_names = list(results.keys())
accuracies  = [results[n]["accuracy"] * 100 for n in model_names]
f1_scores   = [results[n]["f1"] * 100       for n in model_names]
recalls     = [results[n]["recall"] * 100   for n in model_names]
precisions  = [results[n]["precision"] * 100 for n in model_names]

# 10a. Grouped bar chart -- Accuracy, F1, Recall, Precision
x = np.arange(len(model_names))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - 1.5*width, accuracies,  width, label="Accuracy",  color="#3498db")
bars2 = ax.bar(x - 0.5*width, f1_scores,   width, label="F1-Score",  color="#2ecc71")
bars3 = ax.bar(x + 0.5*width, recalls,     width, label="Recall",    color="#e74c3c")
bars4 = ax.bar(x + 1.5*width, precisions,  width, label="Precision", color="#9b59b6")

for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8, fontweight="bold")

ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("Model Performance Comparison - Heart Disease Prediction", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0, 105)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("model_performance_comparison.png", dpi=150)
print("\n  Saved: model_performance_comparison.png")

# 10b. Individual metric bar charts -- Accuracy, F1, Recall
metrics_to_plot = {
    "Accuracy (%)": accuracies,
    "F1-Score (%)": f1_scores,
    "Recall (%)":   recalls,
}
colors = ["#3498db", "#2ecc71", "#e74c3c"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (metric_name, values), color in zip(axes, metrics_to_plot.items(), colors):
    bars = ax.bar(model_names, values, color=color, edgecolor="white", linewidth=1.2)
    ax.set_title(metric_name, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold")
plt.suptitle("Heart Disease Prediction - Model Metrics", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("model_metrics_individual.png", dpi=150, bbox_inches="tight")
print("  Saved: model_metrics_individual.png")

# 10c. Confusion matrices side-by-side
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    r = results[name]
    ax.set_title(f"{name}\nAcc={r['accuracy']*100:.1f}%  "
                 f"F1={r['f1']*100:.1f}%  Rec={r['recall']*100:.1f}%",
                 fontsize=10)
plt.suptitle("Confusion Matrices - Heart Disease Prediction", fontsize=14, y=1.05)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
print("  Saved: confusion_matrices.png")

# 10d. Feature importances (Random Forest)
fig, ax = plt.subplots(figsize=(10, 6))
feat_imp.plot(kind="barh", ax=ax, color="#2ecc71", edgecolor="white")
ax.set_xlabel("Importance", fontsize=12)
ax.set_title("Random Forest - Feature Importances", fontsize=14)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=150)
print("  Saved: feature_importances.png")

print("\n" + "=" * 65)
print("  DONE - All models trained and evaluated successfully!")
print("=" * 65)

import joblib
joblib.dump(models["Logistic Regression"], "lr_model.joblib")
joblib.dump(models["Decision Tree"], "dt_model.joblib")
joblib.dump(models["Random Forest"], "rf_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(list(X.columns), "columns.joblib")