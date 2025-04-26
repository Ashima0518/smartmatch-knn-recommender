import os
import urllib.request
import zipfile
import pandas as pd
import logging
from typing import Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('data_loader')

# MovieLens dataset URLs
MOVIELENS_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_FULL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"

def download_movielens_data(dataset_size: str = 'small', 
                           data_dir: str = './data') -> str:
    """
    Download the MovieLens dataset.
    
    Args:
        dataset_size: 'small' for the small dataset or 'full' for the full dataset
        data_dir: Directory to save the dataset
        
    Returns:
        Path to the dataset directory
    """
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Select URL based on dataset size
    if dataset_size == 'small':
        url = MOVIELENS_SMALL_URL
        dataset_name = 'ml-latest-small'
    else:
        url = MOVIELENS_FULL_URL
        dataset_name = 'ml-latest'
    
    # Path to save the zip file
    zip_path = os.path.join(data_dir, f'{dataset_name}.zip')
    dataset_path = os.path.join(data_dir, dataset_name)
    
    # Check if dataset already exists
    if os.path.exists(dataset_path):
        logger.info(f"Dataset already exists at {dataset_path}")
        return dataset_path
    
    # Download the dataset
    logger.info(f"Downloading MovieLens {dataset_size} dataset from {url}")
    urllib.request.urlretrieve(url, zip_path)
    
    # Extract the dataset
    logger.info(f"Extracting dataset to {data_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Remove the zip file
    os.remove(zip_path)
    
    logger.info(f"Dataset downloaded and extracted to {dataset_path}")
    return dataset_path

def load_movielens_data(dataset_path: Optional[str] = None, 
                       dataset_size: str = 'small',
                       data_dir: str = './data') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the MovieLens dataset.
    
    Args:
        dataset_path: Path to the dataset directory
        dataset_size: 'small' for the small dataset or 'full' for the full dataset
        data_dir: Directory to save the dataset
        
    Returns:
        Tuple of (users, movies, ratings) DataFrames
    """
    # Download the dataset if path not provided
    if dataset_path is None:
        dataset_path = download_movielens_data(dataset_size, data_dir)
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    # Load ratings
    ratings_path = os.path.join(dataset_path, 'ratings.csv')
    logger.info(f"Loading ratings from {ratings_path}")
    ratings = pd.read_csv(ratings_path)
    
    # Load movies
    movies_path = os.path.join(dataset_path, 'movies.csv')
    logger.info(f"Loading movies from {movies_path}")
    movies = pd.read_csv(movies_path)
    
    # Create a simple users DataFrame from unique user IDs
    logger.info("Creating users DataFrame")
    user_ids = ratings['userId'].unique()
    users = pd.DataFrame({'userId': user_ids})
    
    logger.info(f"Loaded {len(users)} users, {len(movies)} movies, and {len(ratings)} ratings")
    return users, movies, ratings

def split_train_test(ratings: pd.DataFrame, test_size: float = 0.2, 
                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the ratings into training and testing sets.
    
    Args:
        ratings: DataFrame with user-item ratings
        test_size: Fraction of ratings to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_ratings, test_ratings) DataFrames
    """
    # Shuffle the ratings
    ratings_shuffled = ratings.sample(frac=1, random_state=random_state)
    
    # Calculate the split index
    split_idx = int(len(ratings) * (1 - test_size))
    
    # Split the data
    train_ratings = ratings_shuffled.iloc[:split_idx]
    test_ratings = ratings_shuffled.iloc[split_idx:]
    
    logger.info(f"Split ratings into {len(train_ratings)} training and {len(test_ratings)} testing samples")
    return train_ratings, test_ratings 