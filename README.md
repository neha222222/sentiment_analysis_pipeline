# Google Play Store Review Sentiment Analysis

A comprehensive ML pipeline for sentiment analysis of Google Play Store reviews, comparing three different approaches: rule-based (VADER/TextBlob), traditional ML (Logistic Regression + TF-IDF), and transformer-based (DistilBERT).

## Project Overview

This project demonstrates core AI/ML engineering skills through:
- Data collection of ~50,000 Google Play Store reviews from Pokémon GO
- Implementation of three sentiment analysis approaches
- Comprehensive evaluation with proper train/test methodology (70/30 split)
- Detailed performance comparison and analysis

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd sentiment-analysis-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (required for preprocessing):
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

## Usage

### Complete Pipeline
Run the entire pipeline from data collection to evaluation:
```bash
python run_pipeline.py
```

Or use the modular approach:
```bash
python src/train_models.py
```

### Individual Components

#### Data Collection
```bash
python src/data_collection.py
```

#### Model Training and Evaluation
```bash
python src/train_models.py --skip-collection
```

## Project Structure

```
sentiment-analysis-project/
├── src/
│   ├── data_collection.py      # Google Play Store scraping
│   ├── preprocessing.py        # Data cleaning and preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_models.py  # VADER and TextBlob
│   │   ├── ml_model.py        # Logistic Regression + TF-IDF
│   │   └── transformer_model.py # DistilBERT
│   ├── evaluation.py          # Metrics and evaluation
│   └── train_models.py        # Main training pipeline
├── data/
│   ├── raw/                   # Raw scraped reviews
│   └── processed/             # Cleaned and preprocessed data
├── results/
│   ├── models/                # Trained model artifacts
│   ├── metrics/               # Evaluation results
│   └── visualizations/        # Charts and plots
├── notebooks/
│   └── exploratory_analysis.ipynb # Data exploration and visualization
├── requirements.txt
└── README.md
```

## Models Implemented

### 1. Baseline Models (Rule-based)
- **VADER**: Valence Aware Dictionary and sEntiment Reasoner
- **TextBlob**: Simple API for diving into common NLP tasks

### 2. Traditional ML Model
- **Logistic Regression + TF-IDF**: Classic machine learning approach with term frequency features

### 3. Transformer Model
- **DistilBERT**: Distilled version of BERT for efficient sentiment classification

## Results Summary

Based on evaluation of 13,482 test reviews from Pokémon GO:

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|-------|----------|-------------------|----------------|------------------|
| VADER | 57.23% | 48.28% | 47.89% | 45.74% |
| TextBlob | 46.57% | 50.53% | 47.97% | 43.20% |
| Logistic Regression + TF-IDF | 71.31% | **60.66%** | **61.76%** | **60.05%** |
| DistilBERT | **78.60%** | 53.70% | 55.14% | 53.79% |

### 🏆 Best Performing Model: Logistic Regression + TF-IDF
- **Highest F1-Score (Macro)**: 60.05% - Best balanced performance across all sentiment classes
- **Strong Accuracy**: 71.31% - Reliable overall prediction accuracy
- **Balanced Precision/Recall**: 60.66%/61.76% - Consistent performance across metrics

## Key Findings

### Complete Analysis Results

1. **Best Performing Model**: Logistic Regression + TF-IDF (F1-Macro: 60.05%)
   - **Why it wins**: Best balanced performance across all sentiment classes despite lower raw accuracy than DistilBERT
   - **Strengths**: Consistent precision/recall, handles class imbalance well, computationally efficient
   - **Use Case**: Ideal for production deployment due to speed and balanced performance

2. **DistilBERT Performance Analysis**:
   - **Highest Accuracy**: 78.60% but lower F1-macro (53.79%)
   - **Issue**: Struggles with minority classes (neutral sentiment) due to class imbalance
   - **Training**: Limited to 2,000 samples and 2 epochs due to computational constraints
   - **Potential**: Could improve with more training data and longer training time

3. **Baseline Model Insights**:
   - **VADER vs TextBlob**: VADER (45.74% F1) outperforms TextBlob (43.20% F1)
   - **Limitation**: Rule-based approaches struggle with gaming app-specific language
   - **Neutral Classification**: Both fail on neutral sentiment (only 8.7% of dataset)

4. **Dataset Characteristics Impact**:
   - **Class Imbalance**: 57% negative, 34% positive, 9% neutral reviews
   - **Quality**: 89.9% data retention after preprocessing (44,937/50,000 reviews)
   - **Domain-Specific**: Gaming app reviews contain unique vocabulary and expressions

5. **Technical Optimization Results**:
   - **TF-IDF**: max_features=10,000, ngram_range=(1,2) optimal
   - **Logistic Regression**: C=1.0 regularization prevents overfitting
   - **Cross-Validation**: 60.23% score confirms model generalization

## Technical Details

### Dataset
- **Source**: Google Play Store reviews for Pokémon GO
- **Size**: ~50,000 reviews collected via google-play-scraper
- **Preprocessing**: Text cleaning, tokenization, sentiment labeling
- **Split**: 70% training (31,455 reviews), 30% testing (13,482 reviews)

### Evaluation Metrics
- **Accuracy**: Overall prediction correctness
- **Precision/Recall/F1**: Per-class and macro-averaged performance
- **Confusion Matrix**: Detailed classification breakdown

### Preprocessing Pipeline
1. Remove NaN values and empty reviews
2. Filter valid rating scores (1-5 stars)
3. Remove very short reviews (< 3 characters)
4. Convert ratings to sentiment labels:
   - 1-2 stars → Negative
   - 3 stars → Neutral  
   - 4-5 stars → Positive
5. Text cleaning and tokenization

## Error Analysis

### Key Challenges Identified
1. **Class Imbalance**: Neutral sentiment severely underrepresented (8.7% of dataset)
2. **DistilBERT Limitations**: Failed to predict any neutral samples (0% precision/recall)
3. **Domain-Specific Language**: Gaming app reviews contain unique vocabulary patterns
4. **Computational Constraints**: DistilBERT trained on limited sample (2,000 reviews, 2 epochs)

### Model-Specific Issues
- **VADER/TextBlob**: Struggle with informal gaming language and abbreviations
- **Logistic Regression**: Best balanced performance but still weak on neutral class
- **DistilBERT**: High accuracy but poor minority class detection

## Future Improvements

### Immediate Enhancements
- **Address Class Imbalance**: Implement SMOTE or class weighting for better neutral sentiment detection
- **DistilBERT Optimization**: Train on full dataset with more epochs and hyperparameter tuning
- **Ensemble Methods**: Combine Logistic Regression + DistilBERT for improved performance

### Advanced Features
- **Domain Adaptation**: Fine-tune models specifically for gaming app review language
- **Real-time Pipeline**: Deploy best model as REST API for live sentiment analysis
- **Multi-language Support**: Extend to non-English reviews using multilingual models
- **Temporal Analysis**: Track sentiment trends over time for app updates/events

### Production Considerations
- **Model Monitoring**: Implement drift detection and performance tracking
- **A/B Testing**: Compare model versions in production environment
- **Scalability**: Optimize for high-throughput batch processing

## License

MIT License
