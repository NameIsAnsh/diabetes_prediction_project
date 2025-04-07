#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exploratory Data Analysis for Diabetes Prediction Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')

# Create output directory for visualizations
output_dir = Path('visualization')
output_dir.mkdir(exist_ok=True)

# Load the dataset
data_path = Path('data/diabetes.csv')
df = pd.read_csv(data_path)

# Display basic information about the dataset
print("Dataset Information:")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# Check for zero values in columns where zeros are not physiologically possible
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in zero_columns:
    zero_count = (df[column] == 0).sum()
    print(f"Number of zeros in {column}: {zero_count} ({zero_count/len(df)*100:.2f}%)")

# Distribution of the target variable
outcome_counts = df['Outcome'].value_counts()
print("\nTarget Variable Distribution:")
print(f"Non-diabetic (0): {outcome_counts[0]} ({outcome_counts[0]/len(df)*100:.2f}%)")
print(f"Diabetic (1): {outcome_counts[1]} ({outcome_counts[1]/len(df)*100:.2f}%)")

# Create visualizations

# 1. Distribution of target variable
plt.figure(figsize=(10, 6))
sns.countplot(x='Outcome', data=df, palette=['#66b3ff', '#ff9999'])
plt.title('Distribution of Diabetes Outcome', fontsize=16)
plt.xlabel('Outcome (0: Non-diabetic, 1: Diabetic)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.savefig(output_dir / 'target_distribution.png', dpi=300, bbox_inches='tight')

# 2. Distribution of features
plt.figure(figsize=(16, 12))
for i, column in enumerate(df.columns[:-1]):
    plt.subplot(3, 3, i+1)
    sns.histplot(data=df, x=column, hue='Outcome', kde=True, palette=['#66b3ff', '#ff9999'])
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.savefig(output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')

# 3. Correlation heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
mask = np.triu(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', mask=mask)
plt.title('Correlation Matrix', fontsize=16)
plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')

# 4. Box plots for each feature by outcome
plt.figure(figsize=(16, 12))
for i, column in enumerate(df.columns[:-1]):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x='Outcome', y=column, data=df, palette=['#66b3ff', '#ff9999'])
    plt.title(f'{column} by Outcome')
plt.tight_layout()
plt.savefig(output_dir / 'boxplots_by_outcome.png', dpi=300, bbox_inches='tight')

# 5. Pairplot for key features
key_features = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'Outcome']
plt.figure(figsize=(16, 12))
sns.pairplot(df[key_features], hue='Outcome', palette=['#66b3ff', '#ff9999'])
plt.savefig(output_dir / 'pairplot_key_features.png', dpi=300, bbox_inches='tight')

# Save the exploratory analysis summary to a file
with open(output_dir / 'eda_summary.txt', 'w') as f:
    f.write("# Exploratory Data Analysis Summary\n\n")
    f.write(f"Dataset Shape: {df.shape}\n\n")
    
    f.write("## Statistical Summary\n")
    f.write(df.describe().to_string())
    f.write("\n\n")
    
    f.write("## Missing Values\n")
    f.write("The dataset doesn't have explicit missing values, but contains zeros in columns where zeros are not physiologically possible:\n")
    for column in zero_columns:
        zero_count = (df[column] == 0).sum()
        f.write(f"- {column}: {zero_count} zeros ({zero_count/len(df)*100:.2f}%)\n")
    f.write("\n")
    
    f.write("## Target Variable Distribution\n")
    f.write(f"- Non-diabetic (0): {outcome_counts[0]} ({outcome_counts[0]/len(df)*100:.2f}%)\n")
    f.write(f"- Diabetic (1): {outcome_counts[1]} ({outcome_counts[1]/len(df)*100:.2f}%)\n\n")
    
    f.write("## Key Observations\n")
    f.write("1. The dataset is imbalanced with more non-diabetic than diabetic cases.\n")
    f.write("2. Several features contain zero values which likely represent missing data.\n")
    f.write("3. Glucose, BMI, and Age show the strongest correlation with the Outcome variable.\n")
    f.write("4. There are outliers in several features that may need to be addressed during preprocessing.\n")

print("\nExploratory data analysis completed. Visualizations saved to the 'visualization' directory.")
