import os
import time
import pandas as pd
from google_play_scraper import app, reviews, Sort
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlayStoreReviewCollector:
    def __init__(self, app_id='com.nianticlabs.pokemongo', target_count=50000):
        self.app_id = app_id
        self.target_count = target_count
        self.collected_reviews = []
        
    def get_app_info(self):
        try:
            app_info = app(self.app_id, lang='en', country='us')
            logger.info(f"App: {app_info['title']}")
            logger.info(f"Total reviews available: {app_info['reviews']}")
            logger.info(f"Rating: {app_info['score']}")
            return app_info
        except Exception as e:
            logger.error(f"Error fetching app info: {e}")
            return None
    
    def collect_reviews(self, batch_size=200, sleep_time=1):
        logger.info(f"Starting collection of {self.target_count} reviews for {self.app_id}")
        
        continuation_token = None
        collected_count = 0
        
        with tqdm(total=self.target_count, desc="Collecting reviews") as pbar:
            while collected_count < self.target_count:
                try:
                    remaining = min(batch_size, self.target_count - collected_count)
                    
                    result, continuation_token = reviews(
                        self.app_id,
                        lang='en',
                        country='us',
                        sort=Sort.NEWEST,
                        count=remaining,
                        continuation_token=continuation_token
                    )
                    
                    if not result:
                        logger.warning("No more reviews available")
                        break
                    
                    self.collected_reviews.extend(result)
                    collected_count += len(result)
                    pbar.update(len(result))
                    
                    logger.info(f"Collected {collected_count}/{self.target_count} reviews")
                    
                    if continuation_token is None:
                        logger.warning("No continuation token, stopping collection")
                        break
                    
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    logger.error(f"Error collecting reviews: {e}")
                    time.sleep(5)
                    continue
        
        logger.info(f"Collection completed. Total reviews collected: {len(self.collected_reviews)}")
        return self.collected_reviews
    
    def save_reviews(self, output_path):
        if not self.collected_reviews:
            logger.error("No reviews to save")
            return False
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df_data = []
        for review in self.collected_reviews:
            df_data.append({
                'reviewId': review.get('reviewId', ''),
                'userName': review.get('userName', ''),
                'content': review.get('content', ''),
                'score': review.get('score', 0),
                'thumbsUpCount': review.get('thumbsUpCount', 0),
                'reviewCreatedVersion': review.get('reviewCreatedVersion', ''),
                'at': review.get('at', ''),
                'replyContent': review.get('replyContent', ''),
                'repliedAt': review.get('repliedAt', '')
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)
        logger.info(f"Reviews saved to {output_path}")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Score distribution:\n{df['score'].value_counts().sort_index()}")
        
        return True

def main():
    collector = PlayStoreReviewCollector()
    
    app_info = collector.get_app_info()
    if not app_info:
        logger.error("Failed to get app info")
        return
    
    reviews_data = collector.collect_reviews()
    
    if reviews_data:
        output_path = 'data/raw/pokemon_go_reviews.csv'
        collector.save_reviews(output_path)
    else:
        logger.error("No reviews collected")

if __name__ == "__main__":
    main()
