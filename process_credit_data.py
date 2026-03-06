import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import pickle
import os

# Load datasets
app = pd.read_csv("application_record.csv")
credit = pd.read_csv("credit_record.csv")

# Data merging and mapping
status_map = {'X': -2, 'C': -1, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
credit['STATUS'] = credit['STATUS'].map(status_map)
credit['TARGET'] = credit['STATUS'].apply(lambda x: 1 if x >= 2 else 0)
target_df = credit.groupby('ID')['TARGET'].max().reset_index()

app['AGE'] = abs(app['DAYS_BIRTH']) / 365
app['YEARS_EMPLOYED'] = abs(app['DAYS_EMPLOYED']) / 365
app.loc[app['DAYS_EMPLOYED'] > 0, 'YEARS_EMPLOYED'] = 0

df = app.merge(target_df, on='ID', how='inner')

# EDA and Preprocessing
df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna('Unknown')

# Feature selection (following the notebook's logic)
features = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 
            'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 
            'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'AGE', 'YEARS_EMPLOYED', 'OCCUPATION_TYPE']
X = df[features]
y = df['TARGET']

# Encoding categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE for class imbalance
sm = SMOTE(sampling_strategy=0.7, random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

# Model Training - RandomForest (common choice for this dataset)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)

# Evaluation
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Export Model and Scaler
with open('credit_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Generate Visualizations for the report
plt.figure(figsize=(10, 6))
sns.countplot(x='TARGET', data=df)
plt.title('Distribution of Target Variable')
plt.savefig('target_distribution.png')

plt.figure(figsize=(12, 6))
sns.histplot(df['AGE'], bins=30, kde=True)
plt.title('Age Distribution of Applicants')
plt.savefig('age_distribution.png')

plt.figure(figsize=(12, 6))
sns.boxplot(x='TARGET', y='AMT_INCOME_TOTAL', data=df)
plt.title('Income vs Target')
plt.ylim(0, 600000) # Limit for better visibility
plt.savefig('income_vs_target.png')

# Feature Importance
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Top 10 Feature Importances')
plt.savefig('feature_importance.png')

# Save summary stats for report
summary_stats = df.describe()
summary_stats.to_csv('summary_stats.csv')

print("Process completed successfully.")
