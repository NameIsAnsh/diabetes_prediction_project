#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Preprocessing for Diabetes Prediction Dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib

# Load the dataset
data_path = Path('data/diabetes.csv')
df = pd.read_csv(data_path)

print("Original dataset shape:", df.shape)

# Identify columns with zero values that likely represent missing data
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace zeros with NaN in these columns
for column in zero_columns:
    df[column] = df[column].replace(0, np.nan)

# Print missing value counts after replacement
print("\nMissing values after zero replacement:")
print(df.isnull().sum())

# Create X (features) and y (target)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining set shape: {X_train.shape}, {y_train.shape}")
print(f"Testing set shape: {X_test.shape}, {y_test.shape}")

# Create a preprocessing pipeline

# 1. Impute missing values with median
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# 2. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Convert back to DataFrames for easier handling
X_train_processed = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_processed = pd.DataFrame(X_test_scaled, columns=X.columns)

print("\nPreprocessed data statistics:")
print(X_train_processed.describe().round(2))

# Save the preprocessed data and preprocessing objects
output_dir = Path('models')
output_dir.mkdir(exist_ok=True)

# Save the preprocessed datasets
np.save(output_dir / 'X_train.npy', X_train_scaled)
np.save(output_dir / 'X_test.npy', X_test_scaled)
np.save(output_dir / 'y_train.npy', y_train.values)
np.save(output_dir / 'y_test.npy', y_test.values)

# Save the preprocessing objects for later use
joblib.dump(imputer, output_dir / 'imputer.pkl')
joblib.dump(scaler, output_dir / 'scaler.pkl')

# Save feature names for reference
joblib.dump(X.columns.tolist(), output_dir / 'feature_names.pkl')

print("\nPreprocessing completed. Preprocessed data and objects saved to 'models' directory.")
