import os
import pandas as pd
import requests
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles downloading and loading of the fake news dataset.
    Uses a publicly available fake news dataset for demonstration.
    """
    
    # Direct URL to a public fake news CSV dataset (Fake or Real News)
    DATA_URL = "https://s3.amazonaws.com/assets.datacamp.com/blog_assets/fake_or_real_news.csv"
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Default to a 'data' dir in the project root
            base_path = Path(__file__).resolve().parent.parent
            self.data_dir = os.path.join(base_path, 'data')
        else:
            self.data_dir = data_dir
            
        os.makedirs(self.data_dir, exist_ok=True)
        self.file_path = os.path.join(self.data_dir, "dataset.csv")

    def download_data(self):
        """Downloads the dataset if it doesn't already exist."""
        if os.path.exists(self.file_path):
            logger.info(f"Dataset already exists at {self.file_path}. Skipping download.")
            return

        logger.info(f"Downloading dataset from {self.DATA_URL}...")
        try:
            response = requests.get(self.DATA_URL, stream=True)
            response.raise_for_status()
            
            with open(self.file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Download complete.")
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise

    def load_data(self) -> pd.DataFrame:
        """
        Loads the dataset into a pandas DataFrame.
        Returns a DataFrame with standardized columns: 'text' and 'label'.
        Label is 1 for FAKE, 0 for REAL.
        """
        self.download_data()
        
        logger.info("Loading dataset into memory...")
        try:
            df = pd.read_csv(self.file_path)
            
            # The specific dataset has columns 'title', 'text', 'label'
            # We will use the 'text' column for training and map labels.
            # 'label' column has values 'FAKE' and 'REAL'
            
            if 'text' not in df.columns or 'label' not in df.columns:
                raise ValueError("Dataset missing required 'text' or 'label' columns.")
                
            # Drop empty texts
            df = df.dropna(subset=['text', 'label'])
            
            # Map labels: FAKE -> 1, REAL -> 0
            df['target'] = df['label'].apply(lambda x: 1 if str(x).upper() == 'FAKE' else 0)
            
            # Select relevant columns
            processed_df = df[['text', 'target']].copy()
            logger.info(f"Loaded {len(processed_df)} records.")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_data()
    print(df.head())
