import unittest
import pandas as pd
import numpy as np
from smartmatch import SmartMatch

class TestSmartMatch(unittest.TestCase):
    """Test cases for the SmartMatch recommendation engine."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample users
        self.users = pd.DataFrame({
            'userId': [1, 2, 3, 4, 5]
        })
        
        # Create sample movies
        self.movies = pd.DataFrame({
            'movieId': [101, 102, 103, 104, 105],
            'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
            'genres': ['Action', 'Comedy', 'Drama', 'Thriller', 'Romance']
        })
        
        # Create sample ratings
        self.ratings = pd.DataFrame({
            'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            'movieId': [101, 102, 103, 101, 103, 105, 102, 103, 104, 101, 102, 105, 103, 104, 105],
            'rating': [5.0, 4.0, 3.0, 4.0, 5.0, 3.0, 3.0, 5.0, 4.0, 5.0, 2.0, 4.0, 2.0, 5.0, 4.0],
            'timestamp': [1000000000] * 15
        })
    
    def test_initialization(self):
        """Test that the recommender initializes correctly."""
        recommender = SmartMatch(n_neighbors=3, metric='cosine', mode='user')
        self.assertEqual(recommender.n_neighbors, 3)
        self.assertEqual(recommender.metric, 'cosine')
        self.assertEqual(recommender.mode, 'user')
    
    def test_fit(self):
        """Test the fit method."""
        recommender = SmartMatch(n_neighbors=2, min_ratings=1)
        recommender.fit(self.users, self.movies, self.ratings)
        
        # Check that matrices were created
        self.assertIsNotNone(recommender.user_item_matrix)
        self.assertIsNotNone(recommender.item_user_matrix)
        
        # Check dimensions
        self.assertEqual(recommender.user_item_matrix.shape, (5, 5))
        self.assertEqual(recommender.item_user_matrix.shape, (5, 5))
        
        # Check that indices were created
        self.assertEqual(len(recommender.user_indices), 5)
        self.assertEqual(len(recommender.item_indices), 5)
    
    def test_user_based_recommendations(self):
        """Test user-based recommendations."""
        recommender = SmartMatch(n_neighbors=2, metric='cosine', mode='user', min_ratings=1)
        recommender.fit(self.users, self.movies, self.ratings)
        
        # Get recommendations for user 1
        recommendations = recommender.recommend(user_id=1, n_recommendations=2)
        
        # Check that we got recommendations
        self.assertGreater(len(recommendations), 0)
        self.assertLessEqual(len(recommendations), 2)
        
        # Check format of recommendations
        for movie, score in recommendations:
            self.assertIsInstance(movie, str)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 5.0)
    
    def test_item_based_recommendations(self):
        """Test item-based recommendations."""
        recommender = SmartMatch(n_neighbors=2, metric='cosine', mode='item', min_ratings=1)
        recommender.fit(self.users, self.movies, self.ratings)
        
        # Get recommendations for user 1
        recommendations = recommender.recommend(user_id=1, n_recommendations=2)
        
        # Check that we got recommendations
        self.assertGreater(len(recommendations), 0)
        self.assertLessEqual(len(recommendations), 2)
        
        # Check format of recommendations
        for movie, score in recommendations:
            self.assertIsInstance(movie, str)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 5.0)
    
    def test_similar_items(self):
        """Test finding similar items."""
        recommender = SmartMatch(n_neighbors=2, metric='cosine', mode='item', min_ratings=1)
        recommender.fit(self.users, self.movies, self.ratings)
        
        # Get similar items
        similar_items = recommender.get_similar_items(item_id=101, n=2)
        
        # Check that we got similar items
        self.assertGreater(len(similar_items), 0)
        self.assertLessEqual(len(similar_items), 2)
        
        # Check format of similar items
        for movie, similarity in similar_items:
            self.assertIsInstance(movie, str)
            self.assertIsInstance(similarity, float)
            self.assertGreaterEqual(similarity, 0.0)
            self.assertLessEqual(similarity, 1.0)
    
    def test_evaluation(self):
        """Test the evaluation method."""
        recommender = SmartMatch(n_neighbors=2, metric='cosine', mode='user', min_ratings=1)
        recommender.fit(self.users, self.movies, self.ratings)
        
        # Create test ratings (a subset of the original ratings)
        test_ratings = self.ratings.sample(frac=0.3, random_state=42)
        
        # Evaluate the model
        metrics = recommender.evaluate(test_ratings, k=2)
        
        # Check that we got metrics
        self.assertIn('precision@k', metrics)
        self.assertIn('recall@k', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('users_evaluated', metrics)

if __name__ == '__main__':
    unittest.main() 