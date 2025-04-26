# SmartMatch: KNN-Based Movie Recommendation Engine

A personalized recommendation system using the K-Nearest Neighbors algorithm to suggest movies based on user preferences and similarities.

## Features

- Uses collaborative filtering based on KNN algorithm
- Provides personalized movie recommendations
- Includes both user-based and item-based recommendation approaches
- Visualization of user/item similarities
- Built-in evaluation metrics

## Installation

```bash
git clone https://github.com/yourusername/smartmatch-knn-recommender.git
cd smartmatch-knn-recommender
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from smartmatch import SmartMatch, load_movielens_data

# Load the MovieLens dataset
users, movies, ratings = load_movielens_data()

# Initialize and train the recommendation model
recommender = SmartMatch(n_neighbors=10)
recommender.fit(users, movies, ratings)

# Get recommendations for a specific user
user_id = 42
recommendations = recommender.recommend(user_id, n_recommendations=10)
print(f"Top recommendations for user {user_id}:")
for movie, score in recommendations:
    print(f"- {movie} (score: {score:.2f})")
```

### Running the Demo

```bash
python demo.py
```

## Demo Results

When running the basic demo with default settings, the system:

1. Downloads and processes the MovieLens small dataset (610 users, 9742 movies, 100,836 ratings)
2. Trains a user-based collaborative filtering model with 10 neighbors using cosine similarity
3. Generates personalized movie recommendations

Example output for a random user (ID 539):

```
Top 10 recommendations for user 539:
1. Treasure Planet (2002) (score: 5.00)
2. Taxi Driver (1976) (score: 5.00)
3. Dances with Wolves (1990) (score: 5.00)
4. Fargo (1996) (score: 5.00)
5. Ghost in the Shell (Kôkaku kidôtai) (1995) (score: 5.00)
6. Platoon (1986) (score: 5.00)
7. Alien (1979) (score: 5.00)
8. Full Metal Jacket (1987) (score: 5.00)
9. Akira (1988) (score: 5.00)
10. Being John Malkovich (1999) (score: 5.00)
```

### Additional Demo Options

- `python demo.py --mode item` - Use item-based collaborative filtering
- `python demo.py --user-id 42` - Generate recommendations for a specific user
- `python demo.py --movie-id 1` - Find similar movies to a specific movie
- `python demo.py --visualize` - Display data visualizations and statistics
- `python demo.py --evaluate` - Evaluate model performance with metrics

## Dataset

This implementation uses the MovieLens dataset, a popular benchmark for recommendation systems. The dataset contains movie ratings from users on a scale of 1-5.

## How It Works

SmartMatch implements two main recommendation approaches:

1. **User-based collaborative filtering**: Recommends items based on similar users' preferences
2. **Item-based collaborative filtering**: Recommends items similar to those the user has liked

The K-Nearest Neighbors algorithm finds the most similar users or items based on rating patterns and uses these similarities to generate recommendations.

## License

MIT 