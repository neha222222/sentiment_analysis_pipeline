import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

from .data_collection import PlayStoreReviewCollector
from .preprocessing import ReviewPreprocessor
from .models.baseline_models import VADERModel, TextBlobModel
from .models.ml_model import LogisticRegressionTFIDFModel
from .models.transformer_model import DistilBERTModel
from .evaluation import ModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentAnalysisPipeline:
    def __init__(self, target_reviews=50000, use_sample_for_bert=True, bert_sample_size=5000):
        self.target_reviews = target_reviews
        self.use_sample_for_bert = use_sample_for_bert
        self.bert_sample_size = bert_sample_size
        self.evaluator = ModelEvaluator()
        
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('results/models', exist_ok=True)
    
    def collect_data(self):
        logger.info("Starting data collection...")
        collector = PlayStoreReviewCollector(target_count=self.target_reviews)
        
        app_info = collector.get_app_info()
        if not app_info:
            raise Exception("Failed to get app info")
        
        reviews = collector.collect_reviews()
        if not reviews:
            raise Exception("Failed to collect reviews")
        
        output_path = 'data/raw/pokemon_go_reviews.csv'
        success = collector.save_reviews(output_path)
        if not success:
            raise Exception("Failed to save reviews")
        
        logger.info(f"Data collection completed. {len(reviews)} reviews saved.")
        return output_path
    
    def preprocess_data(self, input_path):
        logger.info("Starting data preprocessing...")
        
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} reviews from {input_path}")
        
        preprocessor = ReviewPreprocessor()
        processed_df = preprocessor.preprocess_reviews(df)
        
        X_train, X_test, y_train, y_test = preprocessor.create_train_test_split(processed_df)
        
        preprocessor.save_processed_data(X_train, X_test, y_train, y_test, 'data/processed')
        
        return X_train, X_test, y_train, y_test
    
    def train_baseline_models(self, X_test, y_test):
        logger.info("Training baseline models...")
        
        vader_model = VADERModel()
        textblob_model = TextBlobModel()
        
        logger.info("Evaluating VADER model...")
        vader_predictions = vader_model.predict(X_test['content'].values)
        vader_probabilities = vader_model.predict_proba(X_test['content'].values)
        self.evaluator.evaluate_model("VADER", y_test.values, vader_predictions, vader_probabilities)
        
        logger.info("Evaluating TextBlob model...")
        textblob_predictions = textblob_model.predict(X_test['content'].values)
        textblob_probabilities = textblob_model.predict_proba(X_test['content'].values)
        self.evaluator.evaluate_model("TextBlob", y_test.values, textblob_predictions, textblob_probabilities)
        
        return vader_model, textblob_model
    
    def train_ml_model(self, X_train, y_train, X_test, y_test):
        logger.info("Training Logistic Regression + TF-IDF model...")
        
        ml_model = LogisticRegressionTFIDFModel()
        ml_model.train(X_train['cleaned_content'].values, y_train.values, use_grid_search=True)
        
        ml_predictions = ml_model.predict(X_test['cleaned_content'].values)
        ml_probabilities = ml_model.predict_proba(X_test['cleaned_content'].values)
        
        self.evaluator.evaluate_model("Logistic Regression + TF-IDF", y_test.values, ml_predictions, ml_probabilities)
        
        ml_model.save_model('results/models/logistic_regression_tfidf.pkl')
        
        return ml_model
    
    def train_transformer_model(self, X_train, y_train, X_test, y_test):
        logger.info("Training DistilBERT model...")
        
        if self.use_sample_for_bert:
            logger.info(f"Using sample of {self.bert_sample_size} reviews for DistilBERT training...")
            sample_indices = np.random.choice(len(X_train), size=min(self.bert_sample_size, len(X_train)), replace=False)
            X_train_sample = X_train.iloc[sample_indices]
            y_train_sample = y_train.iloc[sample_indices]
            
            val_size = min(1000, len(X_test))
            val_indices = np.random.choice(len(X_test), size=val_size, replace=False)
            X_val = X_test.iloc[val_indices]
            y_val = y_test.iloc[val_indices]
        else:
            X_train_sample = X_train
            y_train_sample = y_train
            X_val = None
            y_val = None
        
        transformer_model = DistilBERTModel()
        transformer_model.train(
            X_train_sample['cleaned_content'].values,
            y_train_sample.values,
            X_val['cleaned_content'].values if X_val is not None else None,
            y_val.values if y_val is not None else None,
            epochs=2,
            batch_size=16
        )
        
        test_sample_size = min(2000, len(X_test))
        test_indices = np.random.choice(len(X_test), size=test_sample_size, replace=False)
        X_test_sample = X_test.iloc[test_indices]
        y_test_sample = y_test.iloc[test_indices]
        
        transformer_predictions = transformer_model.predict(X_test_sample['cleaned_content'].values)
        transformer_probabilities = transformer_model.predict_proba(X_test_sample['cleaned_content'].values)
        
        self.evaluator.evaluate_model("DistilBERT", y_test_sample.values, transformer_predictions, transformer_probabilities)
        
        transformer_model.save_model('results/models/distilbert')
        
        return transformer_model
    
    def run_complete_pipeline(self, skip_collection=False):
        logger.info("Starting complete sentiment analysis pipeline...")
        
        if not skip_collection:
            raw_data_path = self.collect_data()
        else:
            raw_data_path = 'data/raw/pokemon_go_reviews.csv'
            if not os.path.exists(raw_data_path):
                logger.error("Raw data not found. Please run data collection first.")
                return
        
        X_train, X_test, y_train, y_test = self.preprocess_data(raw_data_path)
        
        baseline_models = self.train_baseline_models(X_test, y_test)
        
        ml_model = self.train_ml_model(X_train, y_train, X_test, y_test)
        
        transformer_model = self.train_transformer_model(X_train, y_train, X_test, y_test)
        
        logger.info("Generating evaluation results...")
        self.evaluator.save_all_results()
        
        comparison_df = self.evaluator.create_comparison_table()
        logger.info("\nFinal Model Comparison:")
        logger.info(comparison_df.to_string(index=False))
        
        best_model = max(self.evaluator.results.keys(), key=lambda x: self.evaluator.results[x]['f1_macro'])
        logger.info(f"\nBest performing model: {best_model}")
        logger.info(f"F1-Score (Macro): {self.evaluator.results[best_model]['f1_macro']:.4f}")
        
        logger.info("Pipeline completed successfully!")
        
        return {
            'baseline_models': baseline_models,
            'ml_model': ml_model,
            'transformer_model': transformer_model,
            'evaluator': self.evaluator,
            'best_model': best_model
        }

def main():
    parser = argparse.ArgumentParser(description='Sentiment Analysis Pipeline')
    parser.add_argument('--skip-collection', action='store_true', help='Skip data collection step')
    parser.add_argument('--target-reviews', type=int, default=50000, help='Number of reviews to collect')
    parser.add_argument('--bert-sample-size', type=int, default=5000, help='Sample size for DistilBERT training')
    parser.add_argument('--no-bert-sample', action='store_true', help='Use full dataset for DistilBERT')
    
    args = parser.parse_args()
    
    pipeline = SentimentAnalysisPipeline(
        target_reviews=args.target_reviews,
        use_sample_for_bert=not args.no_bert_sample,
        bert_sample_size=args.bert_sample_size
    )
    
    try:
        results = pipeline.run_complete_pipeline(skip_collection=args.skip_collection)
        logger.info("All tasks completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
