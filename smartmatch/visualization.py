import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union, Tuple

def plot_rating_distribution(ratings_df: pd.DataFrame, title: str = 'Rating Distribution') -> None:
    """
    Plot the distribution of ratings.
    
    Args:
        ratings_df: DataFrame containing ratings
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x='rating', data=ratings_df)
    plt.title(title)
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_user_activity(ratings_df: pd.DataFrame, top_n: int = 20, 
                      title: str = 'Most Active Users') -> None:
    """
    Plot the most active users based on the number of ratings.
    
    Args:
        ratings_df: DataFrame containing ratings
        top_n: Number of top users to show
        title: Title for the plot
    """
    user_counts = ratings_df['userId'].value_counts().reset_index()
    user_counts.columns = ['userId', 'count']
    user_counts = user_counts.sort_values('count', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='userId', y='count', data=user_counts)
    plt.title(title)
    plt.xlabel('User ID')
    plt.ylabel('Number of Ratings')
    plt.xticks(rotation=90)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_movie_popularity(ratings_df: pd.DataFrame, movies_df: pd.DataFrame, 
                         top_n: int = 20, title: str = 'Most Popular Movies') -> None:
    """
    Plot the most popular movies based on the number of ratings.
    
    Args:
        ratings_df: DataFrame containing ratings
        movies_df: DataFrame containing movie information
        top_n: Number of top movies to show
        title: Title for the plot
    """
    movie_counts = ratings_df['movieId'].value_counts().reset_index()
    movie_counts.columns = ['movieId', 'count']
    movie_counts = movie_counts.sort_values('count', ascending=False).head(top_n)
    
    # Get movie titles
    movie_titles = movie_counts.merge(movies_df[['movieId', 'title']], on='movieId')
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='count', y='title', data=movie_titles)
    plt.title(title)
    plt.xlabel('Number of Ratings')
    plt.ylabel('Movie Title')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_rating_heatmap(user_item_matrix: pd.DataFrame, n_users: int = 10, 
                       n_items: int = 20, title: str = 'User-Item Rating Heatmap') -> None:
    """
    Plot a heatmap of user-item ratings.
    
    Args:
        user_item_matrix: User-item rating matrix
        n_users: Number of users to include
        n_items: Number of items to include
        title: Title for the plot
    """
    # Select a subset of users and items
    subset = user_item_matrix.iloc[:n_users, :n_items]
    
    plt.figure(figsize=(16, 10))
    sns.heatmap(subset, cmap='viridis', annot=True, fmt='.1f', linewidths=0.5)
    plt.title(title)
    plt.xlabel('Movie ID')
    plt.ylabel('User ID')
    plt.tight_layout()
    plt.show()

def plot_similar_items(similar_items: List[Tuple[str, float]], 
                     title: str = 'Similar Items') -> None:
    """
    Plot similar items and their similarity scores.
    
    Args:
        similar_items: List of tuples (item_name, similarity_score)
        title: Title for the plot
    """
    item_names = [item[0] for item in similar_items]
    similarity_scores = [item[1] for item in similar_items]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=similarity_scores, y=item_names)
    plt.title(title)
    plt.xlabel('Similarity Score')
    plt.ylabel('Item Name')
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_recommendation_scores(recommendations: List[Tuple[str, float]], 
                             title: str = 'Recommendation Scores') -> None:
    """
    Plot recommendation scores for items.
    
    Args:
        recommendations: List of tuples (item_name, score)
        title: Title for the plot
    """
    item_names = [item[0] for item in recommendations]
    scores = [item[1] for item in recommendations]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=scores, y=item_names)
    plt.title(title)
    plt.xlabel('Recommendation Score')
    plt.ylabel('Item Name')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_evaluation_metrics(metrics: Dict[str, float], title: str = 'Evaluation Metrics') -> None:
    """
    Plot evaluation metrics.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    metrics_to_plot = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and k != 'users_evaluated'}
    sns.barplot(x=list(metrics_to_plot.keys()), y=list(metrics_to_plot.values()))
    plt.title(title)
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.show() 