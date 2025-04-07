# Diabetes Prediction Model Evaluation Report

## Model Comparison

The following models were trained and evaluated:

| model_name             |   accuracy |   precision |   recall |       f1 |      auc |
|:-----------------------|-----------:|------------:|---------:|---------:|---------:|
| Logistic Regression    |   0.744589 |    0.671875 | 0.530864 | 0.593103 | 0.836296 |
| Random Forest          |   0.748918 |    0.671642 | 0.555556 | 0.608108 | 0.81749  |
| Gradient Boosting      |   0.744589 |    0.671875 | 0.530864 | 0.593103 | 0.833498 |
| Support Vector Machine |   0.744589 |    0.683333 | 0.506173 | 0.58156  | 0.817531 |
| K-Nearest Neighbors    |   0.744589 |    0.652778 | 0.580247 | 0.614379 | 0.803416 |
| Neural Network         |   0.727273 |    0.621622 | 0.567901 | 0.593548 | 0.784938 |

## Best Model Performance

The best performing model was **KNeighborsClassifier** with the following metrics on the test set:

- Accuracy: 0.7273
- Precision: 0.6216
- Recall: 0.5679
- F1 Score: 0.5935
- AUC: 0.7849

## Cross-Validation Results

5-fold cross-validation F1 scores: 0.4932, 0.6053, 0.6486, 0.6349, 0.6857
Mean CV F1 score: 0.6135 (±0.0655)

## Model Interpretation

### Confusion Matrix Analysis

- True Negatives: 127 (84.67% of actual negatives)
- False Positives: 23 (15.33% of actual negatives)
- False Negatives: 36 (44.44% of actual positives)
- True Positives: 45 (55.56% of actual positives)

### Classification Report

```
              precision    recall  f1-score   support

Non-Diabetic       0.78      0.85      0.81       150
    Diabetic       0.66      0.56      0.60        81

    accuracy                           0.74       231
   macro avg       0.72      0.70      0.71       231
weighted avg       0.74      0.74      0.74       231
```

## Conclusion

The model demonstrates good performance in predicting diabetes risk, with an F1 score of 0.5935 and AUC of 0.7849. The precision-recall tradeoff is balanced, making the model suitable for screening purposes.

However, there is room for improvement, particularly in reducing false negatives, which are critical in a medical context. Future work could focus on collecting more data, exploring additional features, or implementing ensemble methods to further improve model performance.