#!/usr/bin/env python
# coding: utf-8

# ## Content Based Movie Recommendation Engine on TMDB Dataset

# <b>About:<b/>
#     
# Recommendation systems are imperative in today's day and age. It not only helps the user make quicker and more personalized decisions but also helps the business draw better conversions. There could be millions, if not billions, of products offered by a business, and it's highly likely that the user might not get what they want. Recommendation systems help organisations bridge this gap.
# There are three main types of recommendation systems:
# 1. Content-based: recommendations generated based on the similarity of content consumed. (e.g., Spotify, Netflix, etc)
# 2. Collaborative Based : Recommendations generated based on the similarity of users. (e.g., Facebook, Instagram, etc)
# 3. Hybrid: Utilizes both the above mentioned approaches. (e.g., most E-commerce websites are now adopting this approach.
# 
# <b>Steps Involved:</b>
# 
# 1. Data Fetching
# 2. Pre-Processing
# 3. Model Building
# 4. A Working Website 
# 5. Deploying on Heroku
# 
# <b>Dataset: </b> https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?resource=download

# In[31]:


#importing the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


# In[32]:


#Loading the dataset
movies = pd.read_csv('tmdb_5000_movies.csv') #first dataframe
credits = pd.read_csv('tmdb_5000_credits.csv')  #second dataframe


# In[33]:


#Birds-Eye view of the first dataframe
movies.head(2)


# In[34]:


#Dimensionality of the first dataframe
movies.shape


# In[35]:


#Data Types Involved in the first dataframe
movies.dtypes


# In[36]:


#Now second dataframe
credits.head()


# In[37]:


#Data Types involved in the second dataframe
credits.dtypes


# In[39]:


#Merging the 2 dataframes on 'title'
movies = movies.merge(credits,on='title')


# In[40]:


#New dataframe
movies.shape


# In[41]:


movies.sample()


# <b>Columns to be removed because they may not contribute to the content based tagging:</b> 
# budget, homepage, id, original_language, original_title, popularity, production_comapany, production_countries, release-date

# In[42]:


#Pertinent Columns 
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[43]:


movies.info()


# In[44]:


#Checking for missing data
movies.isna().sum()


# In[45]:


#Drop the nulls
movies.dropna(inplace=True)


# In[46]:


#Checking...
movies.isna().sum()


# In[66]:


#Checking for duplicates
movies.duplicated().sum()


# In[47]:


#Function that converts list of dictionaries to list
import ast #to convert string of lists to lists (abstract syntax tree module)
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 


# In[48]:


#Function that fetches the name of the director from 'crew'.
def director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 
movies['crew'] = movies['crew'].apply(director)


# In[49]:


#Applying the above function on 'genres','keywords' and 'cast' respectively.
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert)
movies.head(2)


# In[50]:


#Slicing the cast
movies['cast'] = movies['cast'].apply(lambda x:x[0:3])


# In[51]:


#Removing spaces in 'crew','cast','genres','keywords' respectively for accurate tagging.
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])


# In[52]:


#Isolating each keyword in the overview 
movies['overview'] = movies['overview'].apply(lambda x:x.split()) 


# In[53]:


movies.sample()


# In[54]:


#A unified column for tagging 
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[55]:


#Creating a new dataframe having only 3 columns: 'movie_id','title', and 'tags'.
final = movies.drop(columns=['overview','genres','keywords','cast','crew'])
final['tags'] = final['tags'].apply(lambda x: " ".join(x))
final.head()


# <b>Vectorizing</b>
# All the tags will be converted to vectors and then the movies having similar vectors (Closest vectors) will be recommended. Bag-Of-Words technique will be utilized in this process. Alternatively, one can also use TFIDF or word2vec. 

# In[59]:


# Vectorization
from sklearn.feature_extraction.text import CountVectorizer
CV = CountVectorizer(max_features=5000,stop_words='english')


# In[60]:


vector = cv.fit_transform(new['tags']).toarray() #to convert scipy sparse matrix to a numpy array
vector.shape


# In[61]:


#How close are the vectors? (distance is inversely proportional to similarity) 
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)
similarity


# In[63]:


def recommend(movie):
    index = final[final['title'] == movie].index[0] #get the index of the movie
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1]) #sort the movies in the descending order, sorting to be done based on the similarity
    for i in distances[1:6]: 
        print(new.iloc[i[0]].title) #printing similar movies
recommend('Spectre')


# In[64]:


pickle.dump(final,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))

