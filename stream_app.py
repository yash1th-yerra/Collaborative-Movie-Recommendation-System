import streamlit as st
import pandas as pd
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load the movie dataset
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Create a movie index mapping (maps movie titles to indices)
movie_idx = dict(zip(movies['title'], list(movies.index)))

# Function for fuzzy title matching
def movie_finder(title):
    all_titles = movies['title'].tolist()
    closest_match = process.extractOne(title, all_titles)
    return closest_match[0]

# Function to get movie recommendations based on content (using genres)
def get_content_based_recommendations(title_string, n_recommendations=10):
    title = movie_finder(title_string)
    idx = movie_idx[title]

    # Generate a "genre vector" for each movie
    count_vectorizer = CountVectorizer(stop_words='english')
    genre_matrix = count_vectorizer.fit_transform(movies['genres'])

    # Compute cosine similarity matrix based on genres
    cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

    # Get the most similar movies
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:(n_recommendations + 1)]
    similar_movies = [i[0] for i in sim_scores]

    recommended_titles = movies['title'].iloc[similar_movies].tolist()
    return recommended_titles

# Streamlit App Interface
st.title('Movie Recommendation System')

# Input: Movie Title
user_input = st.text_input('Enter a movie title:', 'Jumanji')

# Input: Number of recommendations
n_recommendations = st.slider('Number of recommendations:', 1, 20, 5)

# Function to fetch recommendations from the backend
def display_recommendations(title, n_recommendations_):
    # Fetch movie recommendations based on the input title
    recommendations = get_content_based_recommendations(title, n_recommendations_)

    if recommendations:
        st.write(f"**Top {n_recommendations_} Recommendations**:")
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")
    else:
        st.error("Failed to fetch recommendations. Please try again later.")

# Function to find the closest movie title
def display_closest_match(title):
    closest_match = movie_finder(title)
    st.write(f"Did you mean: **{closest_match}**?")

# When the user submits a movie title
if user_input:
    # Find the closest matching movie title
    display_closest_match(user_input)

    # Fetch movie recommendations based on the input title
    display_recommendations(user_input, n_recommendations)
