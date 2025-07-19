# Sentiment Analysis Model Evaluation Report

Generated on: 2025-07-19 01:24:52

## Model Performance Summary

                       Model Accuracy Precision (Macro) Recall (Macro) F1-Score (Macro) Precision (Weighted) Recall (Weighted) F1-Score (Weighted)
                       VADER   0.5723            0.4828         0.4789           0.4574               0.6638            0.5723              0.5870
                    TextBlob   0.4657            0.5053         0.4797           0.4320               0.6743            0.4657              0.5225
Logistic Regression + TF-IDF   0.7131            0.6066         0.6176           0.6005               0.7733            0.7131              0.7381
                  DistilBERT   0.7990            0.5304         0.5757           0.5519               0.7279            0.7990              0.7615


## Detailed Results by Model

### VADER

- **Accuracy**: 0.5723
- **Precision (Macro)**: 0.4828
- **Recall (Macro)**: 0.4789
- **F1-Score (Macro)**: 0.4574

**Per-Class Performance:**
- Negative:
  - Precision: 0.8253
  - Recall: 0.5036
  - F1-Score: 0.6256
- Neutral:
  - Precision: 0.0788
  - Recall: 0.1351
  - F1-Score: 0.0996
- Positive:
  - Precision: 0.5441
  - Recall: 0.7979
  - F1-Score: 0.6470

### TextBlob

- **Accuracy**: 0.4657
- **Precision (Macro)**: 0.5053
- **Recall (Macro)**: 0.4797
- **F1-Score (Macro)**: 0.4320

**Per-Class Performance:**
- Negative:
  - Precision: 0.7971
  - Recall: 0.4019
  - F1-Score: 0.5343
- Neutral:
  - Precision: 0.1031
  - Recall: 0.4656
  - F1-Score: 0.1688
- Positive:
  - Precision: 0.6156
  - Recall: 0.5717
  - F1-Score: 0.5928

### Logistic Regression + TF-IDF

- **Accuracy**: 0.7131
- **Precision (Macro)**: 0.6066
- **Recall (Macro)**: 0.6176
- **F1-Score (Macro)**: 0.6005

**Per-Class Performance:**
- Negative:
  - Precision: 0.8540
  - Recall: 0.7409
  - F1-Score: 0.7934
- Neutral:
  - Precision: 0.1741
  - Recall: 0.3534
  - F1-Score: 0.2333
- Positive:
  - Precision: 0.7917
  - Recall: 0.7585
  - F1-Score: 0.7748

### DistilBERT

- **Accuracy**: 0.7990
- **Precision (Macro)**: 0.5304
- **Recall (Macro)**: 0.5757
- **F1-Score (Macro)**: 0.5519

**Per-Class Performance:**
- Negative:
  - Precision: 0.8069
  - Recall: 0.9125
  - F1-Score: 0.8564
- Neutral:
  - Precision: 0.0000
  - Recall: 0.0000
  - F1-Score: 0.0000
- Positive:
  - Precision: 0.7842
  - Recall: 0.8146
  - F1-Score: 0.7991

## Best Performing Model

**Logistic Regression + TF-IDF** achieved the highest F1-Score (Macro): 0.6005
