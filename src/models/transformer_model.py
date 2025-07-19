import pandas as pd
import numpy as np
import torch
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_encoder = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.label_encoder[self.labels[idx]]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DistilBERTModel:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.name = "DistilBERT"
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.label_decoder = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def initialize_model(self):
        logger.info("Initializing DistilBERT model...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3
        )
        self.model.to(self.device)
    
    def prepare_datasets(self, X_train, y_train, X_val=None, y_val=None):
        train_dataset = ReviewDataset(
            X_train, y_train, self.tokenizer
        )
        
        val_dataset = None
        if X_val is not None and y_val is not None:
            val_dataset = ReviewDataset(
                X_val, y_val, self.tokenizer
            )
        
        return train_dataset, val_dataset
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=3, batch_size=16):
        if self.tokenizer is None or self.model is None:
            self.initialize_model()
        
        logger.info("Preparing datasets...")
        train_dataset, val_dataset = self.prepare_datasets(X_train, y_train, X_val, y_val)
        
        training_args = TrainingArguments(
            output_dir='./results/models/distilbert_checkpoints',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./results/models/distilbert_logs',
            logging_steps=100,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False,
            save_total_limit=2,
        )
        
        callbacks = []
        if val_dataset:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=callbacks,
        )
        
        logger.info("Starting training...")
        self.trainer.train()
        logger.info("Training completed!")
    
    def predict(self, texts):
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    str(text),
                    truncation=True,
                    padding='max_length',
                    max_length=128,
                    return_tensors='pt'
                )
                
                encoding = {k: v.to(self.device) for k, v in encoding.items()}
                
                outputs = self.model(**encoding)
                prediction = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
                predictions.append(self.label_decoder[prediction])
        
        return np.array(predictions)
    
    def predict_proba(self, texts):
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    str(text),
                    truncation=True,
                    padding='max_length',
                    max_length=128,
                    return_tensors='pt'
                )
                
                encoding = {k: v.to(self.device) for k, v in encoding.items()}
                
                outputs = self.model(**encoding)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
                probabilities.append(probs)
        
        return np.array(probabilities)
    
    def save_model(self, filepath):
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained yet!")
        
        os.makedirs(filepath, exist_ok=True)
        self.model.save_pretrained(filepath)
        self.tokenizer.save_pretrained(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        self.tokenizer = DistilBertTokenizer.from_pretrained(filepath)
        self.model = DistilBertForSequenceClassification.from_pretrained(filepath)
        self.model.to(self.device)
        logger.info(f"Model loaded from {filepath}")

def main():
    logger.info("Testing DistilBERT model...")
    
    try:
        train_df = pd.read_csv('data/processed/train_data.csv')
        logger.info(f"Loaded {len(train_df)} training samples")
    except FileNotFoundError:
        logger.error("Processed data not found. Please run preprocessing first.")
        return
    
    sample_size = min(1000, len(train_df))
    sample_df = train_df.sample(n=sample_size, random_state=42)
    
    model = DistilBERTModel()
    
    X_train = sample_df['cleaned_content'].values
    y_train = sample_df['sentiment'].values
    
    model.train(X_train, y_train, epochs=1, batch_size=8)
    
    test_texts = X_train[:5]
    predictions = model.predict(test_texts)
    probabilities = model.predict_proba(test_texts)
    
    logger.info(f"Sample predictions: {predictions}")
    logger.info(f"Sample probabilities shape: {probabilities.shape}")
    
    model.save_model('results/models/distilbert')

if __name__ == "__main__":
    main()
