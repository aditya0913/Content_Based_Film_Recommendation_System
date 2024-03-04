# Content-Based Movie Recommendation Engine


## About

Recommendation systems play a crucial role in aiding user decisions and enhancing business conversions. This project focuses on building a Content-Based Movie Recommendation Engine using the TMDB dataset. Content-based recommendations are generated based on the similarity of content consumed, making them ideal for platforms like Spotify and Netflix.
Developed and deployed a content-based movie recommendation system utilizing the TMDB dataset. The project underwent meticulous data preprocessing and feature engineering to extract pertinent keywords and movie attributes, ensuring accurate recommendations. Leveraging vector similarity techniques, the system identifies movies with similar content profiles for precise recommendations based on user input. The recommendation model was deployed on an interactive website powered by Streamlit. Users can input a movie title, and the system suggests five movies with content similarity, enhancing the experience with prominently displayed movie posters.

## Steps Involved

1. **Data Fetching:** Retrieving data from the TMDB dataset.
2. **Pre-Processing:** Cleaning and organizing the data for analysis.
3. **Model Building:** Creating a recommendation model based on movie content.
4. **A Working Website:** Implementing the recommendation engine on a website.
5. **Deploying on Heroku:** Hosting the application on the Heroku platform.

## Dataset

The dataset used for this project is sourced from [TMDB Movie Metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?resource=download).

## Code Overview

The code involves the following steps:

1. Loading and merging two datasets: `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`.
2. Data pre-processing, including handling missing values and removing irrelevant columns.
3. Extracting relevant information from columns like 'genres,' 'keywords,' 'cast,' and 'crew.'
4. Vectorizing movie tags using the Bag-Of-Words technique.
5. Utilizing cosine similarity to recommend movies based on their content.

## Usage

1. Clone the repository.
2. Run the Python script to build the recommendation engine.
3. Explore the recommendations for different movies.

## Dependencies

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

## Files

- `movie_list.pkl`: Pickle file containing the final movie list.
- `similarity.pkl`: Pickle file containing the cosine similarity matrix.

Feel free to explore and contribute to enhance the recommendation engine!

