#  PHASE 1: IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For display
pd.set_option("display.max_columns", None)

#  PHASE 2: LOAD & INSPECT DATA
# Load the Dataset
Cleaned_Data = pd.read_csv(
    r'C:\Users\HP\Downloads\Excelerate Document\Cleaned_Preprocessed_Data_ML.csv',
    encoding='latin-1'
)

print("Data Loaded Successfully")
print("Shape:", Cleaned_Data.shape)
print(Cleaned_Data.head())

# Dataset Information
print("\nDataset Info:")
print(Cleaned_Data.info())

# Statistics Summary
print("\nNumerical Summary:")
print(Cleaned_Data.describe())

# Check Missing Values
print("\nMissing Values:")
print(Cleaned_Data.isnull().sum())

#PHASE 3: EDA
# Inspect Categorical Columns
print("\nUnique Status Values:", Cleaned_Data['Status_Updated'].unique())

# Visualize Distribution of Status_Updated
plt.figure(figsize=(10,5))
Cleaned_Data['Status_Updated'].value_counts().plot(kind='bar')
plt.title("Status Updated Distribution")
plt.xticks(rotation=45)
plt.show()

# PHASE 3B: PREPROCESSING
# ADJUSTED FOR ENCODED COLUMNS

# Debug: show exact column names
print("Columns:", Cleaned_Data.columns.tolist())

# Normalize column names to remove stray whitespace / unify formatting
Cleaned_Data.columns = Cleaned_Data.columns.str.strip()

# These columns are expected (already encoded in Excel)
encoded_cols = [
    'Gender_Encoding',
    'ApplyMonth_Encoding',
    'Status_Encoded',
]

# Filter to the columns that actually exist and warn about missing ones
present = [c for c in encoded_cols if c in Cleaned_Data.columns]
missing = [c for c in encoded_cols if c not in Cleaned_Data.columns]
if missing:
    print("Warning: these expected encoded columns are missing:", missing)

# Convert only the present columns to numeric
for col in present:
    Cleaned_Data[col] = pd.to_numeric(Cleaned_Data[col], errors='coerce')

# Drop rows with missing numeric values for the present encoded columns
if present:
    Cleaned_Data.dropna(subset=present, inplace=True)

#convert DateTime to Numeric Columns
# Handle Excel serial dates (e.g., "45316.01758") and regular datetime strings
def parse_mixed_datetime(val):
    try:
        # Try parsing as regular datetime first
        return pd.to_datetime(val)
    except:
        try:
            # If that fails, try Excel serial date format
            # Excel serial dates start from 1/1/1900
            return pd.to_datetime(float(val), unit='D', origin='1899-12-30')
        except:
            return pd.NaT

Cleaned_Data['Learner_SignUpDateTime'] = Cleaned_Data['Learner_SignUpDateTime'].apply(parse_mixed_datetime)

Cleaned_Data['SignUp_Year'] = Cleaned_Data['Learner_SignUpDateTime'].dt.year
Cleaned_Data['SignUp_Month'] = Cleaned_Data['Learner_SignUpDateTime'].dt.month
Cleaned_Data['SignUp_Day'] = Cleaned_Data['Learner_SignUpDateTime'].dt.day
Cleaned_Data['SignUp_Hour'] = Cleaned_Data['Learner_SignUpDateTime'].dt.hour
#Drop Original DateTime Column

Cleaned_Data.drop(columns=['Learner_SignUpDateTime'], inplace=True)

# Label Encoding for Remaining Categorical Columns
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

Cleaned_Data['Opportunity_Category_Enc'] = le.fit_transform(Cleaned_Data['Opportunity_Category'])
Cleaned_Data['Opportunity_Name_Enc'] = le.fit_transform(Cleaned_Data['Opportunity_Name'])
# Drop Original Categorical Columns
Cleaned_Data.drop(columns=['Opportunity_Category', 'Opportunity_Name'], inplace=True)
#Encode Country Column
Cleaned_Data = pd.get_dummies(Cleaned_Data, columns=['Country', 'Country_Unique'], drop_first=True)
# Keep only the required columns
required_features = [
    'ApplyMonth_Encoding',
    'Gender_Encoding',
    'SignUp_Year', 'SignUp_Month', 'SignUp_Day', 'SignUp_Hour',
    'Opportunity_Category_Enc',
    'Opportunity_Name_Enc'
] + [col for col in Cleaned_Data.columns if col.startswith('Country_')]

X = Cleaned_Data[required_features]
y = Cleaned_Data['Status_Encoded']

print("Final Shape of Features:", X.shape)
print("Target Variable Shape:", y.shape)

#   PHASE 4: MODEL TRAINING & EVALUATION

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

print("\nPHASE 4: MODEL TRAINING STARTED")

# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)

#  MODEL 1 — LOGISTIC REGRESSION

log_reg = LogisticRegression(max_iter=500, class_weight='balanced')
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)

print("\nLOGISTIC REGRESSION RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_lr, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_lr, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))


#  MODEL 2 — RANDOM FOREST CLASSIFIER

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight='balanced'
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRANDOM FOREST RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_rf, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_rf, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# CONFUSION MATRIX
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#   FEATURE IMPORTANCE

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).plot(kind='bar', figsize=(12,5))
plt.title("Feature Importance (Random Forest)")
plt.show()

print("\nPHASE 4 COMPLETED SUCCESSFULLY")

#     PHASE 5: MODEL TRAINING

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# 1. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Set:", X_train.shape)
print("Test Set:", X_test.shape)

#  LOGISTIC REGRESSION

log_reg = LogisticRegression(max_iter=2000)
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)

print("\n===== Logistic Regression Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_lr, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_lr, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))

#  DECISION TREE

tree = DecisionTreeClassifier(max_depth=6, random_state=42)
tree.fit(X_train, y_train)

y_pred_dt = tree.predict(X_test)

print("\n Decision Tree Results")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_dt, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_dt, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))

#  RANDOM FOREST

rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\n===== Random Forest Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_rf, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_rf, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# CONFUSION MATRIX VISUALIZATION
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# PHASE 6: MODEL INTERPRETATION & DEPLOYMENT

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# 1. FEATURE IMPORTANCE
importances = rf.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n FEATURE IMPORTANCE ")
print(importance_df)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importance (Random Forest)")
plt.show()

# 2. CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Classification Report
print("\n CLASSIFICATION REPORT")
print(classification_report(y_test, y_pred_rf))

# 3. SAVE TRAINED MODEL
model_path = r"C:\Users\HP\Downloads\Excelerate Document\student_dropout_model.pkl"
joblib.dump(rf, model_path)
print(f"\nModel saved successfully at:\n{model_path}")

# 4. SAVE PREDICTIONS
results_df = X_test.copy()
results_df['Actual'] = y_test
results_df['Predicted'] = y_pred_rf

results_path = r"C:\Users\HP\Downloads\Excelerate Document\model_predictions.csv"
results_df.to_csv(results_path, index=False)
print(f"Predictions saved successfully at:\n{results_path}")

# 5. CREATE DEPLOYMENT FUNCTION

def predict_student_dropout(new_data_dict):
    """
    new_data_dict = {
        'ApplyMonth_Encoding': 5,
        'Gender_Encoding': 1,
        'SignUp_Year': 2023,
        'SignUp_Month': 11,
        'SignUp_Day': 21,
        'SignUp_Hour': 14,
        'Opportunity_Category_Enc': 2,
        'Opportunity_Name_Enc': 7,
        ...country columns...
    }
    """
    # Convert dictionary to DataFrame
    new_df = pd.DataFrame([new_data_dict])

    # Ensure all missing country columns are added
    for col in X.columns:
        if col not in new_df.columns:
            new_df[col] = 0  # fill missing one-hot encoded columns

    new_df = new_df[X.columns]  # match original training column order

    # Load saved model
    loaded_model = joblib.load(model_path)

    prediction = loaded_model.predict(new_df)

    return prediction[0]

print("\nDeployment function created successfully!")

# ============================
# PHASE 6: FEATURE IMPORTANCE
# ============================

import matplotlib.pyplot as plt
import numpy as np

# Check if model supports feature_importances_
if hasattr(rf, "feature_importances_"):

    importances = rf.feature_importances_
    feature_names = X.columns

    # Sort feature importance (ascending)
    indices = np.argsort(importances)

    # --- Plot Feature Importance ---
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importance (Random Forest)", fontsize=16)

    plt.barh(
        range(10),  
        importances[indices][-10:],  
        align='center'
    )

    # Only show top 10 names
    plt.yticks(
        range(10),
        [feature_names[i] for i in indices][-10:]
    )

    plt.xlabel("Importance Score", fontsize=14)
    plt.ylabel("Top 10 Features", fontsize=14)
    plt.tight_layout()
    plt.show()

    # --- Print Top 10 in descending order ---
    print("\nTop 10 Most Important Features:")

    for i in reversed(indices[-10:]):  # last 10 indices = top 10
        print(f"{feature_names[i]}: {importances[i]:.4f}")

else:
    print("This model does NOT support feature_importances_.")
# PHASE 7: HYPERPARAMETER TUNING
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Create the model
rf = RandomForestClassifier(random_state=42)

# Grid Search
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,             # 5-fold cross-validation
    n_jobs=-1,        # Use all CPU cores
    verbose=2
)

# Fit the model
grid_search.fit(X_train, y_train)

# Show best parameters
print("\nBest Parameters Found:")
print(grid_search.best_params_)

# Best model after tuning
best_model = grid_search.best_estimator_


# Evaluate tuned model
y_pred_best = best_model.predict(X_test)

print("\nAccuracy After Tuning:", accuracy_score(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))

# PHASE 8: MODEL COMPARISON
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Predictions
# grid_search is the baseline model from earlier
#Best model is the tuned model
y_pred_baseline = grid_search.predict(X_test)
y_pred_tuned = best_model.predict(X_test)

def print_metrics(name, y_true, y_pred):
    print(f"\n----- {name} -----")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average="weighted"))
    print("Recall:", recall_score(y_true, y_pred, average="weighted"))
    print("F1 Score:", f1_score(y_true, y_pred, average="weighted"))

# Compare models
print_metrics("BASELINE MODEL", y_test, y_pred_baseline)
print_metrics("TUNED MODEL", y_test, y_pred_tuned)

import pickle

# Save the tuned model
with open("student_dropout_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save the feature list for future inference
with open("model_features.pkl", "wb") as f:
    pickle.dump(X_train.columns.tolist(), f)

print("Model and feature list saved successfully!")

import pickle
import pandas as pd

# Load model
model = pickle.load(open("student_dropout_model.pkl", "rb"))

# Load feature list
features = pickle.load(open("model_features.pkl", "rb"))

# Example new learner data (must match feature list)
sample = pd.DataFrame([{
    "Apply_Month": 3,
    "Gender_Encoding": 1,
    "SignUp_Year": 2024,
    "SignUp_Month": 7,
    "SignUp_Day": 21,
    "SignUp_Hour": 14,
    "Opportunity_Category_Enc": 2,
    "Opportunity_Name_Enc": 5,
}])

# Predict
prediction = model.predict(sample.reindex(columns=features, fill_value=0))
prediction
# PHASE 9: EXPORT SCRIPT AS PDF
import pypandoc

def convert_py_to_pdf(input_file, output_file):
    pypandoc.convert_text(
        open(input_file, "r", encoding="utf-8").read(),
        to="pdf",
        format="md",
        outputfile=output_file,
        extra_args=["--standalone"]
    )
    print(f"PDF created successfully: {output_file}")

# Convert Script to PDF
convert_py_to_pdf("Excelerate_Dropoff_Predictive_Model_ Completed.py", "Excelerate_Dropoff_Predictive_Model_ Completed.pdf")
