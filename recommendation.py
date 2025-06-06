import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

# Sample user-item ratings data
ratings_dict = {
    'user_id': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E'],
    'item_id': ['Item1', 'Item2', 'Item3', 'Item1', 'Item4', 'Item2', 'Item3', 'Item1', 'Item4', 'Item2'],
    'rating':  [5, 3, 2, 4, 5, 2, 4, 1, 4, 5]
}

# Create a pandas DataFrame
df = pd.DataFrame(ratings_dict)

# Print sample data
print("Sample Ratings Data:")
print(df)

# Define a Reader object with rating scale
reader = Reader(rating_scale=(1, 5))

# Load the dataset from pandas DataFrame
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# Split dataset into train and test sets
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Use SVD (Matrix Factorization) algorithm
algo = SVD()

# Train the algorithm on the trainset
algo.fit(trainset)

# Predict ratings for the testset
predictions = algo.test(testset)

# Calculate Root Mean Squared Error for evaluation
rmse = accuracy.rmse(predictions)

# Function to get top-N item recommendations for a user
def get_top_n_recommendations(algo, user_id, items, n=3):
    # Predict rating for all items not yet rated by user
    user_items = df[df['user_id'] == user_id]['item_id'].unique()
    items_to_predict = [item for item in items if item not in user_items]

    predictions = [algo.predict(user_id, item) for item in items_to_predict]
    # Sort predictions by estimated rating descending
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n = predictions[:n]
    
    print(f"\nTop {n} recommendations for user {user_id}:")
    for pred in top_n:
        print(f"Item: {pred.iid}, Predicted Rating: {pred.est:.2f}")

# List all unique items
all_items = df['item_id'].unique()

# Get recommendations for a sample user
get_top_n_recommendations(algo, user_id='A', items=all_items, n=3)
