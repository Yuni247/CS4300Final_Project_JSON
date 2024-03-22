from collections import defaultdict, Counter
from helpers.MySQLDatabaseHandler import Book, MySQLDatabaseHandler, db
import json
import math
import numpy as np
from IPython.core.display import HTML

import re
from sklearn.feature_extraction.text import CountVectorizer
import os.path
import pickle


def tokenize(text: str):
    # TODO-2.1
    lst = re.findall(r"[A-Za-z]+", text)
    rlst = []
    for word in lst:
      rlst.append(word.lower())
    return rlst


def tokenize_book_feats(book : Book):
    """Returns a list of tokens contained in an entire transcript.


    Parameters
    ----------
    input_transcript : Book
        class Book(db.Model):
        __tablename__ = "books"
        title = db.Column(db.String(150), nullable=False)
        descript = db.Column(db.String(1900), nullable=False)
        authors = db.Column(db.String(100), nullable=False)
        publisher = db.Column(db.String(50), nullable=False)
        categories = db.Column(db.String(50), nullable=False)
        review_score = db.Column(db.String(15), nullable=False)
        review_count = db.Column(db.String(15), nullable=False)


    Returns
    -------
    Dict[list[str]] --> Book[Feature[Token]]
        A dict of lists of tokens representing each feature of one book.
    """
    features = dict()
    features["categories"] = tokenize(book.categories)
    features["authors"] = tokenize(book.authors)
    features["publisher"] = tokenize(book.publisher)
    features["descript"] = tokenize(book.descript)
    return features


#TODO: categories and authors sometimes have multiple fields; need to account for that


#TODO: null fields




def preprocess(query, n=3):
     
        top_categories = []
        top_authors = []
        top_publisher = []


        books = [book.serialize() for book in Book.query.all()]


        # the labels here are the genres
        # categories_mappings maps each categories to an index, authors_mappings maps each author(s) to an index, publisher_mappings maps each publisher to an index


        categories_mappings = {}
        authors_mappings = {}
        publisher_mappings = {}
        descript_lst = []
        book_num_to_mappings = {}
        # book_num_to_mappings is a dict that maps each book id (generated) to a tuple of categories, authors, publisher
        count = 0


        for book in books:      


            if book['categories'] not in categories_mappings:
                categories_mappings[book['categories']] = len(categories_mappings)


            book_num_to_mappings[count] = (categories_mappings[book['categories']], authors_mappings[book['authors']], publisher_mappings[book['publisher']])
            count += 1


            descript_lst.append(book["descript"])


        vectorizer = CountVectorizer(max_df=0.7, min_df=1)


        #vectors is an array where the rows represent book descripts and the columns are words in the corpus
        #dimension of vectors is 734 x num features
        vectors = np.array(vectorizer.fit_transform(descript_lst).toarray())
        features = vectorizer.get_feature_names()




        # genre_count gives probability of book in that categories P(label = y)
        categories_count, authors_count, publisher_count = np.zeros(len(categories_mappings)), np.zeros(len(authors_mappings)), np.zeros(len(publisher_mappings))
        for book in books:
            index = categories_mappings[books['categories']]
            categories_count[index] += 1
            authors_count[index] += 1
            publisher_count[index] += 1


        for i in range(len(categories_count)):
            categories_count[i] /= len(books)
        for i in range(len(authors_count)):
            authors_count[i] /= len(books)
        for i in range(len(publisher_count)):
            publisher_count[i] /= len(books)




        #First we implement Add-1 smoothing so that we don't get non-zero probabilities
        # vectors = vectors + 1


        total_counts = np.sum(vectors, axis = 0)


        if not os.path.exists('categories_prob.txt'):
            # [label]_word_prob gives probability of word occuring given genre P(X | Y)
            categories_word_prob = np.zeros((len(categories_mappings), len(vectors[0])))
           
            #Now we loop through each of the labels in the corpus and for each of the categories we find the label_word_prob
            for i in range(len(categories_mappings)):
                for word_num in range(len(vectors[0])):
                    sum = 0
                    for row in range(len(vectors)):
                            sum += vectors[row][word_num] if book_num_to_mappings[row][0] == i else 0
                    categories_word_prob[i][word_num] = sum / total_counts[word_num]


            pickle.dump(categories_word_prob, open("categories_prob.txt", 'wb'))


        categories_word_prob = pickle.load(open("categories_prob.txt", 'rb'))


        if not os.path.exists('authors_prob.txt'):
            # [label]_word_prob gives probability of word occuring given [label] P(X | Y)
            authors_word_prob = np.zeros((len(authors_mappings), len(vectors[0])))
           
            #Now we loop through each of the labels in the corpus and for each of the categories we find the label_word_prob
            for i in range(len(authors_mappings)):
                for word_num in range(len(vectors[0])):
                    sum = 0
                    for row in range(len(vectors)):
                            sum += vectors[row][word_num] if book_num_to_mappings[row][1] == i else 0
                    authors_word_prob[i][word_num] = sum / total_counts[word_num]


            pickle.dump(authors_word_prob, open("authors_prob.txt", 'wb'))
        authors_word_prob = pickle.load(open("authors_prob.txt", 'rb'))




        if not os.path.exists('publisher_prob.txt'):
            # [label]_word_prob gives probability of word occuring given genre P(X | Y)
            publisher_word_prob = np.zeros((len(publisher_mappings), len(vectors[0])))
           
            #Now we loop through each of the labels in the corpus and for each of the categories we find the label_word_prob
            for i in range(len(publisher_mappings)):
                for word_num in range(len(vectors[0])):
                    sum = 0
                    for row in range(len(vectors)):
                            sum += vectors[row][word_num] if book_num_to_mappings[row][2] == i else 0
                    publisher_word_prob[i][word_num] = sum / total_counts[word_num]


            pickle.dump(publisher_word_prob, open("publisher_prob.txt", 'wb'))


        publisher_word_prob = pickle.load(open("publisher_prob.txt", 'rb'))






        # Now given a query we can iterate through all the categories and see which categories it is most likely to be present in
        query = query.split()
        total_prob_per_categories = np.ones((len(book_num_to_mappings)))
        for i in range(len(book_num_to_mappings)):
            for word in query:
                if word in features:
                    index = features.index(word)
                    total_prob_per_categories[i] *= categories_word_prob[i][index]
            total_prob_per_categories[i] *= categories_count[i]
       
        most_likely_n = np.argsort(total_prob_per_categories)[::-1]
        reverse_categories_mappings = {value[0]: key for key, value in book_num_to_mappings.items()}
       
        for idx in most_likely_n:
            top_categories.append(reverse_categories_mappings[idx])




        # Now given a query we can iterate through all the authors and see which authors it is most likely to be present in
        query = query.split()
        total_prob_per_authors = np.ones((len(book_num_to_mappings)))
        for i in range(len(book_num_to_mappings)):
            for word in query:
                if word in features:
                    index = features.index(word)
                    total_prob_per_authors[i] *= authors_word_prob[i][index]
            total_prob_per_authors[i] *= authors_count[i]
       
        most_likely_n = np.argsort(total_prob_per_authors)[::-1]
        reverse_authors_mappings = {value[1]: key for key, value in book_num_to_mappings.items()}
       
        for idx in most_likely_n:
            top_authors.append(reverse_authors_mappings[idx])




        # Now given a query we can iterate through all the publisher and see which publisher it is most likely to be present in
        query = query.split()
        total_prob_per_publisher = np.ones((len(book_num_to_mappings)))
        for i in range(len(book_num_to_mappings)):
            for word in query:
                if word in features:
                    index = features.index(word)
                    total_prob_per_publisher[i] *= publisher_word_prob[i][index]
            total_prob_per_publisher[i] *= publisher_count[i]
       
        most_likely_n = np.argsort(total_prob_per_publisher)[::-1]
        reverse_publisher_mappings = {value[2]: key for key, value in book_num_to_mappings.items()}
       
        for idx in most_likely_n:
            top_publisher.append(reverse_publisher_mappings[idx])


        return top_categories[:n], top_authors[:n], top_publisher[:n]




def build_idx_helper(idx_dict, tokenized_books_feats, feature):
    for book, book_index in enumerate(tokenized_books_feats):
        for token in book[feature]:
            if token not in idx_dict:
                idx_dict[token] = {book_index:1}
            else:
                if book_index in idx_dict[token]:
                    idx_dict[token][idx_dict] += 1
                else:
                    idx_dict[token][idx_dict] = 1
    for word in idx_dict:
        lst = []
        for doc_id in idx_dict[word]:
            lst.append((doc_id, idx_dict[word][doc_id]))
        idx_dict[word] = lst
    return idx_dict        


def build_inverted_indexes(tokenized_db_feats):
    # YOUR CODE HERE
    authors_idx = {}
    publisher_idx = {}
    categories_idx = {}
    # go thru each book (dict) in the list
    # output 3 inverted indexes (one for each feat)
    authors_idx = build_idx_helper(authors_idx, tokenized_db_feats, "authors")
    publisher_idx = build_idx_helper(publisher_idx, tokenized_db_feats, "publisher")
    categories_idx = build_idx_helper(categories_idx, tokenized_db_feats, "categories")
    return authors_idx, publisher_idx, categories_idx




def compute_idf(inv_idx, n_docs, min_df=5, max_df_ratio=0.95):
    # YOUR CODE HERE
    idf = {}
    for term in inv_idx:
        df = len(inv_idx[term])
        if df >= min_df:
            if df/n_docs <= max_df_ratio:
                frac = n_docs/(1+df)
                idf[term] = math.log(frac, 2)
    return idf


def compute_doc_norms(index, idf, n_docs):
    # YOUR CODE HERE
    norms = np.zeros((n_docs))
    for word in index:
        if word in idf:
            idf_i = idf[word]
            for j in index[word]:
                tf_ij = j[1]
                norms[j[0]] += (tf_ij*idf_i)**2
    norms = norms**(1/2)
    return norms
   
   


def accumulate_dot_scores(query_word_counts, index, idf):
    # YOUR CODE HERE
    doc_scores = {}
    for word in query_word_counts:
        q_i = query_word_counts[word]
        # print(word in index)
        # print(index)
        if word in index:
            ind = index[word]
            for doc in ind:
                # print(doc)
                doc_id = doc[0]
                freq = doc[1]
                if doc_id in doc_scores:
                    doc_scores[doc_id] += idf[word]*q_i*idf[word]*freq
                else:
                    doc_scores[doc_id] = idf[word]*q_i*idf[word]*freq
           
    return doc_scores




def index_search(query, index, idf, doc_norms, rating_dict, thumbs_dict, score_func=accumulate_dot_scores, tokenizer=tokenize):
    # YOUR CODE HERE
    def sortFirst(item):
        return item[0]
    lst = []
    query_tokens = tokenizer.tokenize(query)
    query_word_counts = {}
    for word in query_tokens:
        if word in query_word_counts:
            query_word_counts[word] += 1
        else:
            query_word_counts[word] = 1
           
    scores = score_func(query_word_counts, index, idf)
    q_norm = 0
    for word in query_word_counts:
        if word in idf:
            q_norm += (query_word_counts[word]*idf[word])**2
    q_norm = q_norm**(1/2)
    for doc_id in scores:
        denom = q_norm*doc_norms[doc_id]
        sc = (scores[doc_id]/denom)*(1+rating_dict[doc_id]*thumbs_dict[doc_id])
        lst.append((sc, doc_id))
    lst.sort(key=sortFirst, reverse=True)
    return lst




def get_responses_from_results(response, results):
    # Take results of index search and get list of attractions
    acc = []
    # print(results)
    for x in results:
        id = x[1]
        acc.append(response[id])
    return acc[:21]




   


