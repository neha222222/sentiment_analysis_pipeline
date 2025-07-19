import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        self.metrics_dir = os.path.join(output_dir, 'metrics')
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        
        self.results = {}
    
    def evaluate_model(self, model_name, y_true, y_pred, y_pred_proba=None):
        logger.info(f"Evaluating {model_name}...")
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred, labels=['negative', 'neutral', 'positive'])
        
        report = classification_report(
            y_true, y_pred, 
            labels=['negative', 'neutral', 'positive'],
            output_dict=True,
            zero_division=0
        )
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        if y_pred_proba is not None:
            metrics['probabilities'] = y_pred_proba
        
        self.results[model_name] = metrics
        
        logger.info(f"{model_name} Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision (macro): {precision:.4f}")
        logger.info(f"  Recall (macro): {recall:.4f}")
        logger.info(f"  F1-Score (macro): {f1:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, model_name, save=True):
        if model_name not in self.results:
            logger.error(f"No results found for {model_name}")
            return
        
        cm = self.results[model_name]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive']
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save:
            filepath = os.path.join(self.viz_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {filepath}")
        
        plt.show()
        plt.close()
    
    def create_comparison_table(self, save=True):
        if not self.results:
            logger.error("No results to compare")
            return None
        
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision (Macro)': f"{metrics['precision_macro']:.4f}",
                'Recall (Macro)': f"{metrics['recall_macro']:.4f}",
                'F1-Score (Macro)': f"{metrics['f1_macro']:.4f}",
                'Precision (Weighted)': f"{metrics['precision_weighted']:.4f}",
                'Recall (Weighted)': f"{metrics['recall_weighted']:.4f}",
                'F1-Score (Weighted)': f"{metrics['f1_weighted']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if save:
            filepath = os.path.join(self.metrics_dir, 'model_comparison.csv')
            comparison_df.to_csv(filepath, index=False)
            logger.info(f"Comparison table saved to {filepath}")
        
        return comparison_df
    
    def plot_model_comparison(self, save=True):
        if not self.results:
            logger.error("No results to compare")
            return
        
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            values = [self.results[model][metric] for model in models]
            
            bars = axes[i].bar(models, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(models)])
            axes[i].set_title(f'{metric_name} Comparison')
            axes[i].set_ylabel(metric_name)
            axes[i].set_ylim(0, 1)
            
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.viz_dir, 'model_comparison.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {filepath}")
        
        plt.show()
        plt.close()
    
    def generate_detailed_report(self, save=True):
        if not self.results:
            logger.error("No results to report")
            return
        
        report_lines = []
        report_lines.append("# Sentiment Analysis Model Evaluation Report\n")
        report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report_lines.append("## Model Performance Summary\n")
        
        comparison_df = self.create_comparison_table(save=False)
        if comparison_df is not None:
            report_lines.append(comparison_df.to_string(index=False))
            report_lines.append("\n")
        
        report_lines.append("## Detailed Results by Model\n")
        
        for model_name, metrics in self.results.items():
            report_lines.append(f"### {model_name}\n")
            report_lines.append(f"- **Accuracy**: {metrics['accuracy']:.4f}")
            report_lines.append(f"- **Precision (Macro)**: {metrics['precision_macro']:.4f}")
            report_lines.append(f"- **Recall (Macro)**: {metrics['recall_macro']:.4f}")
            report_lines.append(f"- **F1-Score (Macro)**: {metrics['f1_macro']:.4f}")
            report_lines.append("")
            
            report_lines.append("**Per-Class Performance:**")
            for class_name in ['negative', 'neutral', 'positive']:
                if class_name in metrics['classification_report']:
                    class_metrics = metrics['classification_report'][class_name]
                    report_lines.append(f"- {class_name.capitalize()}:")
                    report_lines.append(f"  - Precision: {class_metrics['precision']:.4f}")
                    report_lines.append(f"  - Recall: {class_metrics['recall']:.4f}")
                    report_lines.append(f"  - F1-Score: {class_metrics['f1-score']:.4f}")
            report_lines.append("")
        
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['f1_macro'])
        report_lines.append(f"## Best Performing Model\n")
        report_lines.append(f"**{best_model}** achieved the highest F1-Score (Macro): {self.results[best_model]['f1_macro']:.4f}\n")
        
        report_content = "\n".join(report_lines)
        
        if save:
            filepath = os.path.join(self.metrics_dir, 'evaluation_report.md')
            with open(filepath, 'w') as f:
                f.write(report_content)
            logger.info(f"Detailed report saved to {filepath}")
        
        return report_content
    
    def save_all_results(self):
        for model_name in self.results.keys():
            self.plot_confusion_matrix(model_name, save=True)
        
        self.create_comparison_table(save=True)
        self.plot_model_comparison(save=True)
        self.generate_detailed_report(save=True)
        
        logger.info("All evaluation results saved!")

def main():
    logger.info("Testing evaluation module...")
    
    y_true = ['positive', 'negative', 'neutral', 'positive', 'negative'] * 20
    y_pred = ['positive', 'positive', 'neutral', 'positive', 'negative'] * 20
    
    evaluator = ModelEvaluator()
    evaluator.evaluate_model("Test Model", y_true, y_pred)
    evaluator.plot_confusion_matrix("Test Model")
    evaluator.create_comparison_table()

if __name__ == "__main__":
    main()
