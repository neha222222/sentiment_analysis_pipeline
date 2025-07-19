import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogisticRegressionTFIDFModel:
    def __init__(self):
        self.name = "Logistic Regression + TF-IDF"
        self.pipeline = None
        self.label_encoder = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.label_decoder = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    def create_pipeline(self):
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english',
                lowercase=True,
                min_df=2,
                max_df=0.95
            )),
            ('classifier', LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ))
        ])
        return pipeline
    
    def train(self, X_train, y_train, use_grid_search=True):
        logger.info("Training Logistic Regression + TF-IDF model...")
        
        y_train_encoded = [self.label_encoder[label] for label in y_train]
        
        if use_grid_search:
            logger.info("Performing hyperparameter tuning...")
            pipeline = self.create_pipeline()
            
            param_grid = {
                'tfidf__max_features': [5000, 10000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'classifier__C': [0.1, 1.0, 10.0]
            }
            
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=3,
                scoring='f1_macro',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train_encoded)
            self.pipeline = grid_search.best_estimator_
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        else:
            self.pipeline = self.create_pipeline()
            self.pipeline.fit(X_train, y_train_encoded)
        
        logger.info("Training completed!")
    
    def predict(self, X_test):
        if self.pipeline is None:
            raise ValueError("Model not trained yet!")
        
        predictions_encoded = self.pipeline.predict(X_test)
        predictions = [self.label_decoder[pred] for pred in predictions_encoded]
        return np.array(predictions)
    
    def predict_proba(self, X_test):
        if self.pipeline is None:
            raise ValueError("Model not trained yet!")
        
        return self.pipeline.predict_proba(X_test)
    
    def save_model(self, filepath):
        if self.pipeline is None:
            raise ValueError("Model not trained yet!")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.pipeline, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        self.pipeline = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")

def main():
    logger.info("Testing Logistic Regression + TF-IDF model...")
    
    try:
        train_df = pd.read_csv('data/processed/train_data.csv')
        test_df = pd.read_csv('data/processed/test_data.csv')
    except FileNotFoundError:
        logger.error("Processed data not found. Please run preprocessing first.")
        return
    
    model = LogisticRegressionTFIDFModel()
    
    X_train = train_df['cleaned_content'].values
    y_train = train_df['sentiment'].values
    X_test = test_df['cleaned_content'].values[:100]
    
    model.train(X_train, y_train, use_grid_search=False)
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    logger.info(f"Sample predictions: {predictions[:5]}")
    logger.info(f"Sample probabilities shape: {probabilities.shape}")
    
    model.save_model('results/models/logistic_regression_tfidf.pkl')

if __name__ == "__main__":
    main()
