#!/usr/bin/env python
"""
SmartMatch: A KNN-Based Movie Recommendation Engine
Demo script showing the core functionalities
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from smartmatch import SmartMatch, load_movielens_data
from smartmatch.visualization import (
    plot_rating_distribution, 
    plot_user_activity,
    plot_movie_popularity,
    plot_recommendation_scores,
    plot_similar_items,
    plot_evaluation_metrics
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SmartMatch Demo')
    parser.add_argument('--mode', type=str, default='user', choices=['user', 'item'],
                        help='Recommendation mode (user or item based)')
    parser.add_argument('--neighbors', type=int, default=10,
                        help='Number of neighbors for KNN')
    parser.add_argument('--metric', type=str, default='cosine',
                        help='Distance metric for KNN')
    parser.add_argument('--user-id', type=int, default=None,
                        help='User ID to generate recommendations for')
    parser.add_argument('--movie-id', type=int, default=None,
                        help='Movie ID to find similar movies for')
    parser.add_argument('--recommendations', type=int, default=10,
                        help='Number of recommendations to generate')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory to store dataset')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate the model on test data')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize dataset statistics')
    
    return parser.parse_args()

def main():
    """Run the SmartMatch demo."""
    args = parse_args()
    
    print("=" * 80)
    print(f"SmartMatch: KNN-Based Movie Recommendation Engine (Mode: {args.mode})")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    users, movies, ratings = load_movielens_data(data_dir=args.data_dir)
    
    # Data summary
    print(f"\nDataset summary:")
    print(f"  - Users: {len(users)}")
    print(f"  - Movies: {len(movies)}")
    print(f"  - Ratings: {len(ratings)}")
    
    # Visualize dataset statistics if requested
    if args.visualize:
        print("\nVisualizing dataset statistics...")
        plot_rating_distribution(ratings)
        plot_user_activity(ratings)
        plot_movie_popularity(ratings, movies)
    
    # Train the model
    print(f"\nTraining SmartMatch with {args.neighbors} neighbors, {args.metric} metric, {args.mode} mode...")
    recommender = SmartMatch(
        n_neighbors=args.neighbors,
        metric=args.metric,
        mode=args.mode
    )
    recommender.fit(users, movies, ratings)
    
    # If user ID is provided, generate recommendations
    if args.user_id is not None:
        user_id = args.user_id
    else:
        # Pick a random user that has at least 5 ratings
        user_counts = ratings['userId'].value_counts()
        active_users = user_counts[user_counts >= 5].index.tolist()
        user_id = np.random.choice(active_users)
    
    # Generate recommendations
    print(f"\nGenerating recommendations for user {user_id}...")
    recommendations = recommender.recommend(
        user_id=user_id,
        n_recommendations=args.recommendations
    )
    
    if recommendations:
        print(f"\nTop {len(recommendations)} recommendations for user {user_id}:")
        for i, (movie, score) in enumerate(recommendations, 1):
            print(f"{i}. {movie} (score: {score:.2f})")
        
        # Visualize recommendations
        plot_recommendation_scores(recommendations, f"Recommendations for User {user_id}")
    else:
        print(f"No recommendations could be generated for user {user_id}")
    
    # Find similar movies if movie ID is provided
    if args.movie_id is not None:
        movie_id = args.movie_id
        
        # Find the movie in the dataset
        if movie_id in recommender.item_indices:
            movie_title = recommender.item_id_to_name[movie_id]
            print(f"\nFinding similar movies to '{movie_title}' (ID: {movie_id})...")
            
            similar_movies = recommender.get_similar_items(movie_id)
            
            print(f"\nMovies similar to '{movie_title}':")
            for i, (similar_movie, similarity) in enumerate(similar_movies, 1):
                print(f"{i}. {similar_movie} (similarity: {similarity:.2f})")
            
            # Visualize similar movies
            plot_similar_items(similar_movies, f"Movies Similar to '{movie_title}'")
        else:
            print(f"\nMovie ID {movie_id} not found in the dataset")
    
    # Evaluate the model if requested
    if args.evaluate:
        print("\nEvaluating the model...")
        
        # Split data into train and test sets
        train_ratio = 0.8
        test_ratio = 1 - train_ratio
        
        train_ratings, test_ratings = train_test_split(
            ratings, 
            test_size=test_ratio, 
            random_state=42,
            stratify=ratings['userId']
        )
        
        print(f"Training set: {len(train_ratings)} ratings")
        print(f"Testing set: {len(test_ratings)} ratings")
        
        # Train the model on the training set
        train_recommender = SmartMatch(
            n_neighbors=args.neighbors,
            metric=args.metric,
            mode=args.mode
        )
        train_recommender.fit(users, movies, train_ratings)
        
        # Evaluate the model
        metrics = train_recommender.evaluate(test_ratings, k=10)
        
        print("\nEvaluation metrics:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  - {metric}: {value:.4f}")
            else:
                print(f"  - {metric}: {value}")
        
        # Visualize evaluation metrics
        plot_evaluation_metrics(metrics)

if __name__ == "__main__":
    main() 