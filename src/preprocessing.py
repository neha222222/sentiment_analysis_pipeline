import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewPreprocessor:
    def __init__(self):
        self.download_nltk_data()
    
    def download_nltk_data(self):
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
    
    def clean_text(self, text):
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        text = re.sub(r'[^\w\s]', ' ', text)
        
        text = re.sub(r'\s+', ' ', text)
        
        text = text.strip().lower()
        
        return text
    
    def convert_rating_to_sentiment(self, rating):
        if rating in [1, 2]:
            return 'negative'
        elif rating == 3:
            return 'neutral'
        elif rating in [4, 5]:
            return 'positive'
        else:
            return 'neutral'
    
    def preprocess_reviews(self, df):
        logger.info("Starting preprocessing...")
        logger.info(f"Initial dataset shape: {df.shape}")
        
        df = df.copy()
        
        df = df.dropna(subset=['content', 'score'])
        logger.info(f"After removing NaN values: {df.shape}")
        
        df = df[df['content'].str.strip() != '']
        logger.info(f"After removing empty reviews: {df.shape}")
        
        df = df[df['score'].isin([1, 2, 3, 4, 5])]
        logger.info(f"After filtering valid scores: {df.shape}")
        
        df['cleaned_content'] = df['content'].apply(self.clean_text)
        
        df = df[df['cleaned_content'].str.len() > 10]
        logger.info(f"After removing very short reviews: {df.shape}")
        
        df['sentiment'] = df['score'].apply(self.convert_rating_to_sentiment)
        
        logger.info("Sentiment distribution:")
        logger.info(df['sentiment'].value_counts())
        
        return df
    
    def create_train_test_split(self, df, test_size=0.3, random_state=42):
        logger.info("Creating train/test split...")
        
        X = df[['cleaned_content', 'content']]
        y = df['sentiment']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        logger.info("Train set sentiment distribution:")
        logger.info(y_train.value_counts())
        logger.info("Test set sentiment distribution:")
        logger.info(y_test.value_counts())
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train, X_test, y_train, y_test, output_dir):
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        train_df = X_train.copy()
        train_df['sentiment'] = y_train
        train_df.to_csv(f"{output_dir}/train_data.csv", index=False)
        
        test_df = X_test.copy()
        test_df['sentiment'] = y_test
        test_df.to_csv(f"{output_dir}/test_data.csv", index=False)
        
        logger.info(f"Processed data saved to {output_dir}")

def main():
    input_path = 'data/raw/pokemon_go_reviews.csv'
    output_dir = 'data/processed'
    
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} reviews from {input_path}")
    except FileNotFoundError:
        logger.error(f"File not found: {input_path}")
        logger.error("Please run data collection first")
        return
    
    preprocessor = ReviewPreprocessor()
    
    processed_df = preprocessor.preprocess_reviews(df)
    
    X_train, X_test, y_train, y_test = preprocessor.create_train_test_split(processed_df)
    
    preprocessor.save_processed_data(X_train, X_test, y_train, y_test, output_dir)

if __name__ == "__main__":
    main()
