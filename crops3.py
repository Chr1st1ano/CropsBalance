import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. LOAD DATASET
df = pd.read_csv('climate_change_impact_on_agriculture_2024.csv')

# Categorize yield into 3 buckets (Low, Medium, High)
bins = [0, 1.5, 3.5, 6]
labels = ['Low', 'Medium', 'High']
df['Yield_Bucket'] = pd.cut(df['Crop_Yield_MT_per_HA'], bins=bins, labels=labels)

# 2. PREPROCESSING
import joblib
for col in ['Country', 'Region', 'Crop_Type', 'Adaptation_Strategies']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    joblib.dump(le, f'{col}_encoder.pkl')

feature_order = [c for c in df.columns if c not in ['Crop_Yield_MT_per_HA', 'Yield_Bucket']]
joblib.dump(feature_order, 'feature_order.pkl')

X = df.drop(columns=['Crop_Yield_MT_per_HA', 'Yield_Bucket'])
y = df['Yield_Bucket']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. CLASS DISTRIBUTION (BEFORE) ---
print("--- Class Distribution Before Balancing ---")
print(y_train.value_counts())

# --- 4. MANUAL UNDER-SAMPLING (The fix for your error) ---
# We do this using basic pandas so you don't need 'imblearn'
train_data = X_train.copy()
train_data['target'] = y_train

# Find the size of the smallest class (usually 'High')
min_size = train_data['target'].value_counts().min()

# Randomly sample from each group to match the smallest size
balanced_df = pd.concat([
    train_data[train_data['target'] == 'Low'].sample(min_size, random_state=42),
    train_data[train_data['target'] == 'Medium'].sample(min_size, random_state=42),
    train_data[train_data['target'] == 'High'].sample(min_size, random_state=42)
])

X_train_bal = balanced_df.drop(columns=['target'])
y_train_bal = balanced_df['target']

print("\n--- Class Distribution After Balancing ---")
print(y_train_bal.value_counts())

# 5. MODEL TRAINING
# Model 1: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_bal, y_train_bal)

# Model 2: Logistic Regression (Needs scaling)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_bal)
X_test_sc = scaler.transform(X_test)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_sc, y_train_bal)

# 6. EVALUATION
rf_acc = accuracy_score(y_test, rf.predict(X_test))
lr_acc = accuracy_score(y_test, lr.predict(X_test_sc))

print(f"\nRandom Forest Accuracy: {rf_acc:.4f}")
print(f"Logiastic Regression Accuracy: {lr_acc:.4f}")
print(f"Best Model: {'Random Forest' if rf_acc > lr_acc else 'Logistic Regression'}")

# 7. SAVE THE BEST MODEL
import joblib

# Choose the best model (Random Forest performed better)
best_model = rf if rf_acc > lr_acc else lr
scaler_used = scaler if rf_acc <= lr_acc else None

# Save the model
joblib.dump(best_model, 'crop_yield_model.pkl')
if scaler_used:
    joblib.dump(scaler_used, 'crop_yield_scaler.pkl')

print(f"\nBest model saved: {'Random Forest' if rf_acc > lr_acc else 'Logistic Regression'}")

# 8. VISUALIZATION
plt.figure(figsize=(10, 5))
sns.barplot(x=['Random Forest', 'Logistic Regression'], y=[rf_acc, lr_acc])
plt.title('Comparison of Model Performance')
plt.ylabel('Accuracy Score')
plt.show()