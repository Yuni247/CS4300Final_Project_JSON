from __future__ import print_function
import numpy as np
import os
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import random
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')

# Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    books_df = pd.DataFrame(data)

# Create a TF-IDF vectorizer
vectorizer =  TfidfVectorizer(max_df=.7)

# Compute the TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(x['descript'] for x in data)

# do SVD with a very large k (we usually use 100), just for the sake of getting many sorted singular values (aka importances)
#u, s, v_trans = svds(tfidf_matrix, k=100)

docs_compressed, s, words_compressed = svds(tfidf_matrix, k=40)
words_compressed = words_compressed.transpose()

word_to_index = vectorizer.vocabulary_
index_to_word = {i:t for t,i in word_to_index.items()}

# index_to_book = {i:t for t,i in enumerate(data)}
# book_to_index = {t:i for t,i in enumerate(data)}
# cosine similarity
def closest_words(word_in, words_representation_in, k = 10):
    if word_in not in word_to_index: 
        guess = []
        j = 0
        while j < k+1:
          rand = random.randint(0, len(word_to_index))
          guess.append((rand, index_to_word[rand]))
          j += 1
          #print("randomW")
        return [(ind, word, 0) for ind,word in guess]
    sims = words_representation_in.dot(words_representation_in[word_to_index[word_in],:])
    asort = np.argsort(-sims)[:k+1]
    return [(index_to_word[i],sims[i]) for i in asort[1:]]

words_compressed_normed = normalize(words_compressed, axis = 1)
docs_compressed_normed = normalize(docs_compressed)

# this is basically the same cosine similarity code that we used before, just with some changes to
# the returned output format to let us print out the documents in a sensible way
def closest_projects(project_index_in, project_repr_in, k = 5):
    sims = project_repr_in.dot(project_repr_in[project_index_in,:])
    asort = np.argsort(-sims)[:k+1]
    return [(data[i],sims[i]) for i in asort[1:]]

# Once again, basically the same cosine similarity code, but mixing two different matrices
def closest_projects_to_word(word_in, k = 5):
    #print(word_in)
    if word_in not in word_to_index: 
        guess = []
        j = 0
        while j < k+1:
          rand = random.randint(0, len(data))
          guess.append((rand, data[rand]))
          j += 1
        return [(ind, book, 0) for ind,book in guess]
    sims = docs_compressed_normed.dot(words_compressed_normed[word_to_index[word_in],:])
    asort = np.argsort(-sims)[:k+1]
    return [(i, data[i],sims[i]) for i in asort[1:]]

def recommend_book(mood_interest):
  filtered_books = []
  for _, proj, _ in closest_projects_to_word(mood_interest):
    filtered_books.append(proj)
  if filtered_books:
    return filtered_books
  else:
    other_words = closest_words(mood_interest)
    i = 0
    while i < 5:
      for _, proj, _ in closest_projects_to_word(other_words[i]):
        filtered_books.append(proj)
      i += 1
    return filtered_books
