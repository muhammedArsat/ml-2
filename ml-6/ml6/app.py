import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, request, render_template

# Load dataset
dataset_path = 'movies.csv'  # Update this path as needed
df = pd.read_csv('movies (1).csv')

# Preprocess the dataset
if 'title' in df.columns and 'genres' in df.columns:
    df['genres'] = df['genres'].str.replace('|', ' ', regex=False)
    df['combined_features'] = df['genres']

    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    # Compute cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Function to get movie recommendations based on title
    def get_recommendations(title, cosine_sim=cosine_sim):
        if title not in df['title'].values:
            return f"Movie '{title}' not found in the dataset."

        idx = df[df['title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]  # Skip the first score (itself)
        movie_indices = [i[0] for i in sim_scores]
        return df['title'].iloc[movie_indices].tolist()

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['title']
    recommendations = get_recommendations(movie_title)
    return render_template('index.html', title=movie_title, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)