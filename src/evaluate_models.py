#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Evaluation and Performance Analysis for Diabetes Prediction
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
from utils import predict_diabetes 
# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')

# Load data and models
models_dir = Path('models')
results_dir = Path('models/results')
evaluation_dir = Path('models/evaluation')
evaluation_dir.mkdir(exist_ok=True)

# Load preprocessed data
X_train = np.load(models_dir / 'X_train.npy')
X_test = np.load(models_dir / 'X_test.npy')
y_train = np.load(models_dir / 'y_train.npy')
y_test = np.load(models_dir / 'y_test.npy')

feature_names = joblib.load(models_dir / 'feature_names.pkl')

# Load the final model
final_model = joblib.load(models_dir / 'final_model.pkl')

# Load model comparison results
model_comparison = pd.read_csv(results_dir / 'model_comparison.csv')

print("Loaded model comparison results:")
print(model_comparison)

# Create a comprehensive model comparison visualization
plt.figure(figsize=(14, 10))
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
model_names = model_comparison['model_name'].tolist()

# Create a grouped bar chart for all metrics
x = np.arange(len(model_names))
width = 0.15
multiplier = 0

fig, ax = plt.subplots(figsize=(15, 8))

for metric in metrics:
    offset = width * multiplier
    rects = ax.bar(x + offset, model_comparison[metric], width, label=metric.capitalize())
    ax.bar_label(rects, fmt='.2f', padding=3, rotation=90, fontsize=8)
    multiplier += 1

ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(evaluation_dir / 'comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')

# Detailed evaluation of the final model
print(f"\nDetailed evaluation of the final model: {type(final_model).__name__}")

# 1. Cross-validation analysis
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(final_model, X_train, y_train, cv=cv, scoring='f1')

print(f"Cross-validation F1 scores: {cv_scores}")
print(f"Mean CV F1 score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# 2. Learning curve analysis
train_sizes, train_scores, test_scores = learning_curve(
    final_model, X_train, y_train, cv=cv, scoring='f1',
    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42, n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, label='Cross-validation score', color='green', marker='s')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')
plt.xlabel('Training Set Size')
plt.ylabel('F1 Score')
plt.title('Learning Curve Analysis')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(evaluation_dir / 'learning_curve.png', dpi=300, bbox_inches='tight')

# 3. ROC and Precision-Recall curves
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(evaluation_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
plt.axhline(y=sum(y_test)/len(y_test), color='red', linestyle='--', label='No Skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True)
plt.savefig(evaluation_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')

# 4. Confusion Matrix with percentages
y_pred = final_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm_percent = cm / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Diabetic', 'Diabetic'],
            yticklabels=['Non-Diabetic', 'Diabetic'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Count')
plt.savefig(evaluation_dir / 'confusion_matrix_count.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 8))
sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues', 
            xticklabels=['Non-Diabetic', 'Diabetic'],
            yticklabels=['Non-Diabetic', 'Diabetic'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Percentage')
plt.savefig(evaluation_dir / 'confusion_matrix_percent.png', dpi=300, bbox_inches='tight')

# 5. Classification Report
report = classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic'], output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Save the classification report
with open(evaluation_dir / 'classification_report.txt', 'w') as f:
    f.write("Classification Report for Final Model\n\n")
    f.write(classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic']))

# 6. Create a simple prediction function for the interface
def predict_diabetes(features_dict):
    """
    Make a prediction using the final model.
    
    Parameters:
    -----------
    features_dict : dict
        Dictionary with feature names as keys and feature values as values.
        
    Returns:
    --------
    dict
        Dictionary with prediction results.
    """
    # Load preprocessing objects
    imputer = joblib.load(models_dir / 'imputer.pkl')
    scaler = joblib.load(models_dir / 'scaler.pkl')
    
    # Create feature array in the correct order
    features = np.array([[features_dict[feature] for feature in feature_names]])
    
    # Apply preprocessing
    features_imputed = imputer.transform(features)
    features_scaled = scaler.transform(features_imputed)
    
    # Make prediction
    prediction = final_model.predict(features_scaled)[0]
    probability = final_model.predict_proba(features_scaled)[0, 1]
    
    return {
        'prediction': int(prediction),
        'probability': float(probability),
        'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
    }

# Save the prediction function
joblib.dump(predict_diabetes, models_dir / 'predict_function.pkl')


# 7. Create a comprehensive evaluation report
with open(evaluation_dir / 'model_evaluation_report.md', 'w') as f:
    f.write("# Diabetes Prediction Model Evaluation Report\n\n")
    
    f.write("## Model Comparison\n\n")
    f.write("The following models were trained and evaluated:\n\n")
    f.write(model_comparison.to_markdown(index=False))
    f.write("\n\n")
    
    f.write("## Best Model Performance\n\n")
    f.write(f"The best performing model was **{type(final_model).__name__}** with the following metrics on the test set:\n\n")
    
    best_model_metrics = model_comparison.iloc[-1]
    f.write(f"- Accuracy: {best_model_metrics['accuracy']:.4f}\n")
    f.write(f"- Precision: {best_model_metrics['precision']:.4f}\n")
    f.write(f"- Recall: {best_model_metrics['recall']:.4f}\n")
    f.write(f"- F1 Score: {best_model_metrics['f1']:.4f}\n")
    f.write(f"- AUC: {best_model_metrics['auc']:.4f}\n\n")
    
    f.write("## Cross-Validation Results\n\n")
    f.write(f"5-fold cross-validation F1 scores: {', '.join([f'{score:.4f}' for score in cv_scores])}\n")
    f.write(f"Mean CV F1 score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})\n\n")
    
    f.write("## Model Interpretation\n\n")
    f.write("### Confusion Matrix Analysis\n\n")
    f.write(f"- True Negatives: {cm[0, 0]} ({cm_percent[0, 0]:.2%} of actual negatives)\n")
    f.write(f"- False Positives: {cm[0, 1]} ({cm_percent[0, 1]:.2%} of actual negatives)\n")
    f.write(f"- False Negatives: {cm[1, 0]} ({cm_percent[1, 0]:.2%} of actual positives)\n")
    f.write(f"- True Positives: {cm[1, 1]} ({cm_percent[1, 1]:.2%} of actual positives)\n\n")
    
    f.write("### Classification Report\n\n")
    f.write("```\n")
    f.write(classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic']))
    f.write("```\n\n")
    
    f.write("## Conclusion\n\n")
    f.write("The model demonstrates good performance in predicting diabetes risk, with an F1 score of ")
    f.write(f"{best_model_metrics['f1']:.4f} and AUC of {best_model_metrics['auc']:.4f}. ")
    f.write("The precision-recall tradeoff is balanced, making the model suitable for screening purposes.\n\n")
    
    f.write("However, there is room for improvement, particularly in reducing false negatives, ")
    f.write("which are critical in a medical context. Future work could focus on collecting more data, ")
    f.write("exploring additional features, or implementing ensemble methods to further improve model performance.")

print("\nModel evaluation completed. Results saved to 'models/evaluation' directory.")

# Test the prediction function with a sample case
sample_features = {
    'Pregnancies': 6,
    'Glucose': 148,
    'BloodPressure': 72,
    'SkinThickness': 35,
    'Insulin': 0,
    'BMI': 33.6,
    'DiabetesPedigreeFunction': 0.627,
    'Age': 50
}

prediction_result = predict_diabetes(sample_features)
print("\nSample Prediction:")
print(f"Features: {sample_features}")
print(f"Prediction: {'Diabetic' if prediction_result['prediction'] == 1 else 'Non-Diabetic'}")
print(f"Probability: {prediction_result['probability']:.4f}")
print(f"Risk Level: {prediction_result['risk_level']}")
