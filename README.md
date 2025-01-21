# Movie Recommendation System

## Overview
This project demonstrates a content-based movie recommendation system using the following steps:
- Feature extraction from movie data.
- Calculating similarity scores using TF-IDF vectorization and cosine similarity.
- Recommending movies based on user input.

## Dataset
The dataset used for this project is `movies.csv`. It contains the following attributes:
- **title**: Title of the movie.
- **genres**: Genre(s) of the movie.
- **keywords**: Keywords describing the movie.
- **tagline**: Tagline of the movie.
- **cast**: Main cast of the movie.
- **director**: Director of the movie.
- **index**: A unique index for each movie.

## Project Steps

### 1. Data Loading and Exploration
- Load the dataset using Pandas.
- Display the first 5 rows to understand the structure of the data.
- Check the shape of the dataset to get the number of rows and columns.

### 2. Data Preprocessing
- Select relevant features for recommendation: `genres`, `keywords`, `tagline`, `cast`, and `director`.
- Handle missing values by replacing them with empty strings.
- Combine all selected features into a single string for each movie.

### 3. Feature Extraction
- Convert the combined text data into numerical feature vectors using TF-IDF vectorization.
- Compute the similarity scores between movies using cosine similarity.

### 4. Recommendation System
- Take a user’s favorite movie as input.
- Find close matches for the input movie from the dataset using the `difflib` library.
- Retrieve the index of the best match and calculate similarity scores for all movies.
- Recommend the top 30 movies based on similarity scores.

## Code
```python
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
movies_data = pd.read_csv('/content/movies.csv')

# Display the first 5 rows
print(movies_data.head())

# Get the number of rows and columns
print(movies_data.shape)

# Select relevant features
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
print(selected_features)

# Replace missing values with empty strings
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine the selected features
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']
print(combined_features)

# Convert the text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
print(feature_vectors)

# Calculate similarity scores
similarity = cosine_similarity(feature_vectors)
print(similarity)

# User input for favorite movie
movie_name = input('Enter your favourite movie name: ')

# Convert the titles to a list
list_of_all_titles = movies_data['title'].tolist()

# Find close matches for the user's input
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

if find_close_match:
    # Get the best match
    close_match = find_close_match[0]

    # Find the index of the matched movie
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

    # Calculate similarity scores
    similarity_score = list(enumerate(similarity[index_of_the_movie]))

    # Sort movies based on similarity scores
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    print('Movies suggested for you:\n')

    # Display the top 30 similar movies
    i = 1
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        if i < 30:
            print(i, '.', title_from_index)
            i += 1
else:
    print("No close matches found for the movie name you entered. Please try again with a different movie name.")
```

## Dependencies
- Python 3.x
- pandas
- numpy
- sklearn

## How to Run
1. Clone the repository.
2. Install the required libraries using:
   ```bash
   pip install pandas numpy scikit-learn
   ```
3. Place the `movies.csv` file in the appropriate directory.
4. Run the Python script and input your favorite movie when prompted.
