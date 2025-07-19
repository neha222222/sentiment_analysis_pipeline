import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VADERModel:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.name = "VADER"
    
    def predict(self, texts):
        predictions = []
        
        for text in texts:
            if pd.isna(text) or text == '':
                predictions.append('neutral')
                continue
            
            scores = self.analyzer.polarity_scores(str(text))
            compound = scores['compound']
            
            if compound >= 0.05:
                predictions.append('positive')
            elif compound <= -0.05:
                predictions.append('negative')
            else:
                predictions.append('neutral')
        
        return np.array(predictions)
    
    def predict_proba(self, texts):
        probabilities = []
        
        for text in texts:
            if pd.isna(text) or text == '':
                probabilities.append([0.33, 0.34, 0.33])
                continue
            
            scores = self.analyzer.polarity_scores(str(text))
            
            neg_prob = max(0, scores['neg'])
            neu_prob = max(0, scores['neu'])
            pos_prob = max(0, scores['pos'])
            
            total = neg_prob + neu_prob + pos_prob
            if total > 0:
                neg_prob /= total
                neu_prob /= total
                pos_prob /= total
            else:
                neg_prob = neu_prob = pos_prob = 1/3
            
            probabilities.append([neg_prob, neu_prob, pos_prob])
        
        return np.array(probabilities)

class TextBlobModel:
    def __init__(self):
        self.name = "TextBlob"
    
    def predict(self, texts):
        predictions = []
        
        for text in texts:
            if pd.isna(text) or text == '':
                predictions.append('neutral')
                continue
            
            try:
                blob = TextBlob(str(text))
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    predictions.append('positive')
                elif polarity < -0.1:
                    predictions.append('negative')
                else:
                    predictions.append('neutral')
            except:
                predictions.append('neutral')
        
        return np.array(predictions)
    
    def predict_proba(self, texts):
        probabilities = []
        
        for text in texts:
            if pd.isna(text) or text == '':
                probabilities.append([0.33, 0.34, 0.33])
                continue
            
            try:
                blob = TextBlob(str(text))
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    pos_prob = min(0.8, 0.5 + polarity)
                    neg_prob = max(0.1, 0.5 - polarity) / 2
                    neu_prob = 1 - pos_prob - neg_prob
                elif polarity < -0.1:
                    neg_prob = min(0.8, 0.5 - polarity)
                    pos_prob = max(0.1, 0.5 + polarity) / 2
                    neu_prob = 1 - neg_prob - pos_prob
                else:
                    neu_prob = 0.6
                    pos_prob = neg_prob = 0.2
                
                probabilities.append([neg_prob, neu_prob, pos_prob])
            except:
                probabilities.append([0.33, 0.34, 0.33])
        
        return np.array(probabilities)

def main():
    logger.info("Testing baseline models...")
    
    test_texts = [
        "This game is amazing! I love it so much!",
        "This app is terrible and crashes all the time.",
        "It's okay, nothing special but works fine.",
        "",
        "Best game ever created! Highly recommend!"
    ]
    
    vader_model = VADERModel()
    textblob_model = TextBlobModel()
    
    logger.info("VADER predictions:")
    vader_preds = vader_model.predict(test_texts)
    for i, (text, pred) in enumerate(zip(test_texts, vader_preds)):
        logger.info(f"Text {i+1}: '{text[:50]}...' -> {pred}")
    
    logger.info("\nTextBlob predictions:")
    textblob_preds = textblob_model.predict(test_texts)
    for i, (text, pred) in enumerate(zip(test_texts, textblob_preds)):
        logger.info(f"Text {i+1}: '{text[:50]}...' -> {pred}")

if __name__ == "__main__":
    main()
