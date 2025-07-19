import nltk
import logging
from src.data_collection import PlayStoreReviewCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_nltk_data():
    logger.info("Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True) 
    nltk.download('vader_lexicon', quiet=True)
    logger.info("NLTK data downloaded successfully")

def test_data_collection():
    logger.info("Testing data collection with small sample...")
    collector = PlayStoreReviewCollector(target_count=10)
    
    app_info = collector.get_app_info()
    if not app_info:
        logger.error("Failed to get app info")
        return False
    
    logger.info(f"App: {app_info['title']}")
    logger.info(f"Total reviews available: {app_info['reviews']}")
    logger.info(f"Rating: {app_info['score']}")
    
    reviews = collector.collect_reviews()
    if reviews:
        logger.info(f"Successfully collected {len(reviews)} sample reviews")
        return True
    else:
        logger.error("Failed to collect reviews")
        return False

if __name__ == "__main__":
    download_nltk_data()
    success = test_data_collection()
    if success:
        logger.info("Setup test completed successfully!")
    else:
        logger.error("Setup test failed!")
