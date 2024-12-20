from flask import Flask, request, jsonify
import pandas as pd
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Create Flask app
app = Flask(__name__)

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


# Flask routes
@app.route('/')
def home():
    return 'Welcome to the Movie Recommendation API! Use /recommend to get movie recommendations.'


@app.route('/recommend', methods=['GET'])
def recommend_movies():
    title = request.args.get('title', default='Jumanji', type=str)
    n_recommendations = request.args.get('n_recommendations', default=5, type=int)

    recommendations = get_content_based_recommendations(title, n_recommendations)
    return jsonify({
        'input_movie': title,
        'recommended_movies': recommendations
    })


@app.route('/movie_finder', methods=['GET'])
def find_movie():
    title = request.args.get('title', default='Jumanji', type=str)
    closest_match = movie_finder(title)
    return jsonify({
        'input_title': title,
        'closest_match': closest_match
    })


if __name__ == '__main__':
    app.run(debug=True)
