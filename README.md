# Movie Recommendation System

This project implements a movie recommendation system using the **MovieLens dataset**. The system employs different recommendation techniques such as **Collaborative Filtering**, **Content-Based Filtering**, and **Matrix Factorization** to provide personalized movie recommendations based on user preferences.

## Project Overview

The goal of this project is to build a recommendation system that can suggest movies to users based on:

- **Collaborative Filtering**: Recommending movies based on the preferences of similar users.
- **Content-Based Filtering**: Recommending movies based on the movie's content (genres).
- **Matrix Factorization**: Decomposing the user-item interaction matrix to find latent features and uncover hidden patterns.

## Dataset

This project uses the **MovieLens** dataset, which contains:

- **Movie data**: Information about the movies, such as movie ID, title, and genres.
- **Rating data**: Ratings provided by users for each movie.

Dataset files used:
- `movies.csv` — Contains movie information like title and genres.
- `ratings.csv` — Contains user ratings for each movie.

## Requirements

- Python 3.x
- Required libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scipy`
  - `sklearn`
  - `fuzzywuzzy`
  - `python-Levenshtein`

Install the dependencies using `pip`:

```
pip install -r requirements.txt
```


How It Works
Data Preprocessing

The dataset is preprocessed into a user-item interaction matrix (utility matrix) where:

    Rows represent users.
    Columns represent movies.
    Values represent the ratings given by the user to a movie.

The matrix is sparse, as most users rate only a small fraction of movies. This sparsity is evaluated to understand the cold-start problem.
Collaborative Filtering

The core of the collaborative filtering approach is to identify users who share similar preferences and recommend movies that similar users have liked. The nearest neighbor search is done using the k-Nearest Neighbors (k-NN) algorithm.
Content-Based Filtering

Content-based filtering generates recommendations based on the features of the items (in this case, movies). The genres of movies are transformed into binary features. The Cosine Similarity metric is then used to find the most similar movies based on their genres.
Matrix Factorization

Matrix Factorization (e.g., TruncatedSVD) is used to decompose the user-item matrix into two smaller matrices representing users' preferences and items' (movies') characteristics. The decomposed matrices help in predicting missing ratings and suggesting relevant movies.
Cold Start Problem

Collaborative Filtering and Matrix Factorization are sensitive to the cold start problem, which arises when new users or movies with no prior ratings are introduced. This is mitigated by incorporating Content-Based Filtering to recommend movies based on their features, even for users who have interacted with very few movies.
Movie Finder Function

To enhance the user experience, fuzzy string matching is implemented using the fuzzywuzzy library to find the closest matching movie title from user input. This helps in handling typos and partial inputs.
Recommendation Functions

    Collaborative Filtering (k-NN): Returns the top-k most similar movies based on user-item ratings.
    Content-Based Filtering: Returns movies similar to a given movie based on genre similarity.
    Matrix Factorization: Returns the top-k most similar movies using a decomposed user-item interaction matrix.



Live Demo :  https://hybrid-model-movie-recommendation-system-canx7ckjx7mmsnsarysk7.streamlit.app/


