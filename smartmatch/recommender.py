import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Tuple, Dict, Optional, Union

class SmartMatch:
    """
    A KNN-based recommendation engine that uses collaborative filtering
    to provide personalized movie recommendations.
    """
    
    def __init__(self, n_neighbors: int = 5, metric: str = 'cosine', 
                 mode: str = 'user', min_ratings: int = 5):
        """
        Initialize the SmartMatch recommendation engine.
        
        Args:
            n_neighbors: Number of neighbors to consider for recommendations
            metric: Distance metric ('cosine', 'euclidean', etc.)
            mode: 'user' for user-based or 'item' for item-based recommendations
            min_ratings: Minimum number of ratings required for a user/item to be included
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.mode = mode
        self.min_ratings = min_ratings
        self.knn_model = None
        self.user_item_matrix = None
        self.item_user_matrix = None
        self.users_df = None
        self.items_df = None
        self.ratings_df = None
        self.user_indices = {}
        self.item_indices = {}
        self.item_id_to_name = {}
        self.name_to_item_id = {}
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('SmartMatch')
    
    def fit(self, users_df: pd.DataFrame, items_df: pd.DataFrame, 
            ratings_df: pd.DataFrame) -> 'SmartMatch':
        """
        Fit the recommendation model with user, item, and rating data.
        
        Args:
            users_df: DataFrame containing user information
            items_df: DataFrame containing item information
            ratings_df: DataFrame containing user-item ratings
            
        Returns:
            Self instance for method chaining
        """
        self.logger.info("Fitting SmartMatch recommendation model...")
        
        # Store the original dataframes
        self.users_df = users_df
        self.items_df = items_df
        self.ratings_df = ratings_df
        
        # Create item name mappings
        self.item_id_to_name = dict(zip(items_df['movieId'], items_df['title']))
        self.name_to_item_id = {v: k for k, v in self.item_id_to_name.items()}
        
        # Create user-item matrix
        self.logger.info("Creating user-item matrix...")
        user_item_matrix = ratings_df.pivot(index='userId', 
                                           columns='movieId', 
                                           values='rating').fillna(0)
        
        # Filter users and items with enough ratings
        user_ratings_count = (user_item_matrix > 0).sum(axis=1)
        item_ratings_count = (user_item_matrix > 0).sum(axis=0)
        
        filtered_users = user_ratings_count[user_ratings_count >= self.min_ratings].index
        filtered_items = item_ratings_count[item_ratings_count >= self.min_ratings].index
        
        self.user_item_matrix = user_item_matrix.loc[filtered_users, filtered_items]
        self.item_user_matrix = self.user_item_matrix.T
        
        # Create indices mappings
        self.user_indices = {user_id: idx for idx, user_id in enumerate(self.user_item_matrix.index)}
        self.item_indices = {item_id: idx for idx, item_id in enumerate(self.user_item_matrix.columns)}
        
        # Fit the KNN model
        self.logger.info(f"Fitting KNN model in {self.mode}-based mode...")
        if self.mode == 'user':
            matrix = self.user_item_matrix.values
        else:
            matrix = self.item_user_matrix.values
            
        self.knn_model = NearestNeighbors(n_neighbors=self.n_neighbors + 1,  # +1 because item will be its own neighbor
                                          metric=self.metric,
                                          algorithm='auto')
        self.knn_model.fit(matrix)
        
        self.logger.info("SmartMatch model fitting completed successfully")
        return self
    
    def _get_neighbors(self, entity_id: int, entity_type: str = 'user') -> List[Tuple[int, float]]:
        """Get the nearest neighbors for a user or item."""
        if entity_type == 'user':
            if entity_id not in self.user_indices:
                self.logger.warning(f"User {entity_id} not found in the dataset")
                return []
            entity_idx = self.user_indices[entity_id]
            matrix = self.user_item_matrix.values
            entity_ids = list(self.user_indices.keys())
        else:
            if entity_id not in self.item_indices:
                self.logger.warning(f"Item {entity_id} not found in the dataset")
                return []
            entity_idx = self.item_indices[entity_id]
            matrix = self.item_user_matrix.values
            entity_ids = list(self.item_indices.keys())
        
        # Get distances and indices of the nearest neighbors
        distances, indices = self.knn_model.kneighbors([matrix[entity_idx]])
        
        # Skip the first one (it's the entity itself)
        neighbors = [(entity_ids[idx], 1 - dist) for dist, idx in 
                     zip(distances[0][1:], indices[0][1:])]
        
        return neighbors
    
    def recommend(self, user_id: int, n_recommendations: int = 10, 
                 exclude_rated: bool = True) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: The ID of the user to generate recommendations for
            n_recommendations: Number of recommendations to generate
            exclude_rated: Whether to exclude items already rated by the user
            
        Returns:
            List of tuples (movie_title, similarity_score)
        """
        if user_id not in self.user_indices:
            self.logger.warning(f"User {user_id} not in the dataset. Cannot generate recommendations.")
            return []
        
        if self.mode == 'user':
            return self._user_based_recommend(user_id, n_recommendations, exclude_rated)
        else:
            return self._item_based_recommend(user_id, n_recommendations, exclude_rated)
    
    def _user_based_recommend(self, user_id: int, n_recommendations: int, 
                             exclude_rated: bool) -> List[Tuple[str, float]]:
        """Generate recommendations using user-based collaborative filtering."""
        # Get similar users
        similar_users = self._get_neighbors(user_id, 'user')
        
        if not similar_users:
            return []
        
        # Get the user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        
        # Get items the user hasn't rated or has rated low
        if exclude_rated:
            unrated_items = user_ratings[user_ratings == 0].index
        else:
            unrated_items = user_ratings.index
        
        # Calculate predicted ratings for unrated items
        predictions = {}
        
        for item_id in unrated_items:
            item_scores = []
            for sim_user_id, similarity in similar_users:
                rating = self.user_item_matrix.loc[sim_user_id, item_id]
                if rating > 0:  # Only consider actual ratings
                    item_scores.append((rating, similarity))
            
            # Calculate weighted average if we have scores
            if item_scores:
                weighted_sum = sum(rating * similarity for rating, similarity in item_scores)
                sum_similarities = sum(similarity for _, similarity in item_scores)
                if sum_similarities > 0:
                    predictions[item_id] = weighted_sum / sum_similarities
        
        # Sort and convert item IDs to titles
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        recommendations = [(self.item_id_to_name[item_id], score) 
                          for item_id, score in sorted_predictions[:n_recommendations]]
        
        return recommendations
    
    def _item_based_recommend(self, user_id: int, n_recommendations: int, 
                             exclude_rated: bool) -> List[Tuple[str, float]]:
        """Generate recommendations using item-based collaborative filtering."""
        # Get the user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0]
        
        if exclude_rated:
            candidate_items = [item for item in self.item_indices.keys() 
                              if item not in rated_items.index]
        else:
            candidate_items = list(self.item_indices.keys())
        
        # Calculate scores for candidate items
        item_scores = {}
        
        for rated_item_id, rating in rated_items.items():
            # Skip if the item is not in our filtered dataset
            if rated_item_id not in self.item_indices:
                continue
                
            # Get similar items
            similar_items = self._get_neighbors(rated_item_id, 'item')
            
            for sim_item_id, similarity in similar_items:
                if sim_item_id in candidate_items:
                    if sim_item_id not in item_scores:
                        item_scores[sim_item_id] = []
                    item_scores[sim_item_id].append((rating, similarity))
        
        # Calculate weighted averages
        predictions = {}
        for item_id, scores in item_scores.items():
            weighted_sum = sum(rating * similarity for rating, similarity in scores)
            sum_similarities = sum(similarity for _, similarity in scores)
            if sum_similarities > 0:
                predictions[item_id] = weighted_sum / sum_similarities
        
        # Sort and convert item IDs to titles
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        recommendations = [(self.item_id_to_name[item_id], score) 
                          for item_id, score in sorted_predictions[:n_recommendations]]
        
        return recommendations
    
    def get_similar_users(self, user_id: int, n: int = 5) -> List[Tuple[int, float]]:
        """Get users similar to the given user."""
        return self._get_neighbors(user_id, 'user')[:n]
    
    def get_similar_items(self, item_id: int, n: int = 5) -> List[Tuple[str, float]]:
        """Get items similar to the given item."""
        neighbors = self._get_neighbors(item_id, 'item')[:n]
        return [(self.item_id_to_name[neighbor_id], similarity) 
                for neighbor_id, similarity in neighbors]
    
    def evaluate(self, test_ratings: pd.DataFrame, k: int = 10) -> Dict[str, float]:
        """
        Evaluate the recommendation system on test data.
        
        Args:
            test_ratings: DataFrame with user-item ratings for testing
            k: Number of recommendations to generate for each user
            
        Returns:
            Dictionary with evaluation metrics
        """
        precision_sum = 0
        recall_sum = 0
        user_count = 0
        
        unique_users = test_ratings['userId'].unique()
        
        for user_id in unique_users:
            if user_id not in self.user_indices:
                continue
                
            # Get ground truth - items the user rated in test set
            user_test_items = set(test_ratings[test_ratings['userId'] == user_id]['movieId'])
            
            # Get recommendations
            recommendations = self.recommend(user_id, n_recommendations=k)
            if not recommendations:
                continue
                
            # Extract recommended item IDs
            rec_items = []
            for title, _ in recommendations:
                if title in self.name_to_item_id:
                    rec_items.append(self.name_to_item_id[title])
            
            rec_items = set(rec_items)
            
            # Calculate precision and recall
            relevant_recs = rec_items.intersection(user_test_items)
            
            precision = len(relevant_recs) / len(rec_items) if rec_items else 0
            recall = len(relevant_recs) / len(user_test_items) if user_test_items else 0
            
            precision_sum += precision
            recall_sum += recall
            user_count += 1
        
        # Calculate averages
        avg_precision = precision_sum / user_count if user_count > 0 else 0
        avg_recall = recall_sum / user_count if user_count > 0 else 0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        return {
            'precision@k': avg_precision,
            'recall@k': avg_recall,
            'f1_score': f1_score,
            'users_evaluated': user_count
        } 