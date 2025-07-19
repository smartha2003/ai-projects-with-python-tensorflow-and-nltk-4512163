# 📚 Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors  # For finding similar items
from scipy.sparse import csr_matrix             # Efficient storage for sparse user-item matrix

# 📂 Load the ratings and movies data
ratings = pd.read_csv("ratings.csv")  # Contains userId, movieId, and rating
movies = pd.read_csv("movies.csv")    # Contains movieId and title

# 🧱 Create a user-item interaction matrix (sparse)
def create_matrix(df):
    # Number of unique users and movies
    N = len(df['userId'].unique())
    M = len(df['movieId'].unique())

    # Create mappings from IDs to matrix indices and back
    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))

    # Map original user and movie IDs to matrix index positions
    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]

    # Build the sparse matrix where:
    # - rows represent movies
    # - columns represent users
    # - cell values are the ratings
    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))

    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

# 🔧 Create the matrix and mappings
X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)

# 🔍 Find similar movies using k-Nearest Neighbors
def find_similar_movies(movie_id, X, k, metric='cosine', show_distance=False):
    neighbour_ids = []

    # Convert movie ID to matrix index
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]

    # Add 1 to k because the movie itself will be the closest
    k += 1

    # Fit kNN model on the movie vectors
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)

    # Reshape movie vector for input to kNN
    movie_vec = movie_vec.reshape(1, -1)

    # Find k nearest neighbors
    neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)

    # Map matrix indices back to movie IDs
    for i in range(0, k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])

    # Remove the movie itself from the list
    neighbour_ids.pop(0)
    return neighbour_ids

# 🎥 Create a dictionary to map movie IDs to their titles
movie_titles = dict(zip(movies['movieId'], movies['title']))

# 🧪 Test the recommender system with a movie
movie_id = 3  # "Grumpier Old Men (1995)"
similar_ids = find_similar_movies(movie_id, X, k=10)
movie_title = movie_titles[movie_id]

# 🖨️ Display the recommendations
print(f"Since you watched {movie_title}, you might also like:")
for i in similar_ids:
    print(movie_titles[i])