from src.train_models import SentimentAnalysisPipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting complete sentiment analysis pipeline...")
    
    pipeline = SentimentAnalysisPipeline(
        target_reviews=50000,
        use_sample_for_bert=True,
        bert_sample_size=2000
    )
    
    try:
        results = pipeline.run_complete_pipeline(skip_collection=True)
        logger.info("Pipeline completed successfully!")
        
        logger.info("\n" + "="*50)
        logger.info("FINAL RESULTS SUMMARY")
        logger.info("="*50)
        
        comparison_df = results['evaluator'].create_comparison_table(save=False)
        print(comparison_df.to_string(index=False))
        
        logger.info(f"\nBest performing model: {results['best_model']}")
        logger.info(f"F1-Score (Macro): {results['evaluator'].results[results['best_model']]['f1_macro']:.4f}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
