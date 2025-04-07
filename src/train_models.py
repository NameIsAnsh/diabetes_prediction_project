#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train Machine Learning Models for Diabetes Prediction
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel

# Set random seed for reproducibility
np.random.seed(42)

# Load preprocessed data
models_dir = Path('models')
X_train = np.load(models_dir / 'X_train.npy')
X_test = np.load(models_dir / 'X_test.npy')
y_train = np.load(models_dir / 'y_train.npy')
y_test = np.load(models_dir / 'y_test.npy')

# Load feature names
with open(models_dir / 'feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Features: {feature_names}")

# Create output directory for results
results_dir = Path('models/results')
results_dir.mkdir(exist_ok=True)

# Function to evaluate model performance
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate AUC if probability predictions are available
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if auc is not None:
        print(f"AUC: {auc:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Diabetic', 'Diabetic'],
                yticklabels=['Non-Diabetic', 'Diabetic'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(results_dir / f'{model_name.replace(" ", "_").lower()}_confusion_matrix.png', 
                dpi=300, bbox_inches='tight')
    
    # Generate ROC curve if probability predictions are available
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.savefig(results_dir / f'{model_name.replace(" ", "_").lower()}_roc_curve.png', 
                    dpi=300, bbox_inches='tight')
    
    # Save detailed classification report
    report = classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic'])
    with open(results_dir / f'{model_name.replace(" ", "_").lower()}_report.txt', 'w') as f:
        f.write(f"{model_name} Classification Report\n\n")
        f.write(report)
        f.write(f"\nAccuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        if auc is not None:
            f.write(f"AUC: {auc:.4f}\n")
    
    # Save the model
    with open(results_dir / f'{model_name.replace(" ", "_").lower()}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Return metrics for comparison
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'model': model
    }

# Define models to train
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Neural Network': MLPClassifier(max_iter=1000, random_state=42)
}

# Train and evaluate each model
results = []
for name, model in models.items():
    print(f"\nTraining {name}...")
    result = evaluate_model(model, X_train, X_test, y_train, y_test, name)
    results.append(result)

# Find the best model based on F1 score
best_model = max(results, key=lambda x: x['f1'])
print(f"\nBest model based on F1 score: {best_model['model_name']}")
print(f"F1 Score: {best_model['f1']:.4f}")

# Feature importance analysis for tree-based models
if isinstance(best_model['model'], (RandomForestClassifier, GradientBoostingClassifier)):
    # Get feature importances
    importances = best_model['model'].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.title(f'Feature Importances - {best_model["model_name"]}')
    plt.bar(range(X_train.shape[1]), importances[indices], align='center')
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(results_dir / 'feature_importances.png', dpi=300, bbox_inches='tight')
    
    # Save feature importances to file
    with open(results_dir / 'feature_importances.txt', 'w') as f:
        f.write("Feature Importances:\n\n")
        for i in indices:
            f.write(f"{feature_names[i]}: {importances[i]:.4f}\n")

# Hyperparameter tuning for the best model
print("\nPerforming hyperparameter tuning for the best model...")

if best_model['model_name'] == 'Logistic Regression':
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'penalty': ['l1', 'l2']
    }
    best_model_class = LogisticRegression(max_iter=1000, random_state=42)

elif best_model['model_name'] == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    best_model_class = RandomForestClassifier(random_state=42)

elif best_model['model_name'] == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5]
    }
    best_model_class = GradientBoostingClassifier(random_state=42)

elif best_model['model_name'] == 'Support Vector Machine':
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'linear']
    }
    best_model_class = SVC(probability=True, random_state=42)

elif best_model['model_name'] == 'K-Nearest Neighbors':
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }
    best_model_class = KNeighborsClassifier()

else:  # Neural Network
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
    best_model_class = MLPClassifier(max_iter=1000, random_state=42)

# Perform grid search
grid_search = GridSearchCV(best_model_class, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate the tuned model
tuned_model = grid_search.best_estimator_
tuned_result = evaluate_model(tuned_model, X_train, X_test, y_train, y_test, 
                             f"{best_model['model_name']} (Tuned)")

# Compare original and tuned model
print("\nModel Comparison:")
print(f"Original {best_model['model_name']} F1 Score: {best_model['f1']:.4f}")
print(f"Tuned {best_model['model_name']} F1 Score: {tuned_result['f1']:.4f}")
print(f"Improvement: {(tuned_result['f1'] - best_model['f1']) * 100:.2f}%")

# Save the final model
final_model = tuned_model
with open(models_dir / 'final_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

# Save model comparison results
model_comparison = pd.DataFrame(results)
model_comparison = model_comparison.drop('model', axis=1)
model_comparison.to_csv(results_dir / 'model_comparison.csv', index=False)

# Create a bar chart comparing model performance
plt.figure(figsize=(12, 8))
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
model_names = [result['model_name'] for result in results]

for i, metric in enumerate(metrics):
    values = [result[metric] for result in results]
    plt.subplot(2, 3, i+1)
    sns.barplot(x=model_names, y=values)
    plt.title(f'{metric.capitalize()} Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()

plt.savefig(results_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')

print("\nModel training and evaluation completed. Results saved to 'models/results' directory.")
