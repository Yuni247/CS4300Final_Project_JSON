from collections import defaultdict, Counter
#from helpers.MySQLDatabaseHandler import Book, MySQLDatabaseHandler, db
import json
import math
import numpy as np
#from IPython.core.display import HTML

import re
#from sklearn.feature_extraction.text import CountVectorizer
#import os.path
#import pickle


def tokenize(text: str):
    # TODO-2.1
    lst = re.findall(r"[A-Za-z]+", text)
    rlst = []
    for word in lst:
      rlst.append(word.lower())
    return rlst

def tokenize_authors(text: str):
    # Remove the square brackets and single quotes from the string
    cleaned_text = text.replace("[", "").replace("]", "").replace("'", "")
    
    # Split the string into individual authors based on the comma separator
    authors = cleaned_text.split(", ")
    
    # Convert each author name to lowercase
    authors = [author.lower() for author in authors]
    
    return authors


def process_books_df(books_df):
    print("STARTED PROCESSING DATABASE")
    processed_data = []

    # Iterate through each row in the DataFrame
    for index, row in books_df.iterrows():
        authors = tokenize_authors(row['authors'])
        descript = tokenize(row['descript'])
        categories = tokenize(row['categories'])

        # Store the tokenized data in a dictionary
        processed_record = {
            'authors': authors,
            'descript': descript,
            'categories': categories
        }

        processed_data.append(processed_record)
    print("FINISHED PROCESSING DATABASE")
    return processed_data


def build_idx_helper(idx_dict, tokenized_books_feats, feature):
    # for book_idx, book in enumerate(tokenized_books_feats):
    #     for token in book[feature]:
    #         if token not in idx_dict:
    #             idx_dict[token] = {book_idx:1}
    #         else:
    #             if book_idx in idx_dict[token]:
    #                 idx_dict[token][book_idx] += 1
    #             else:
    #                 idx_dict[token][book_idx] = 1
    # for word in idx_dict:
    #     lst = []
    #     for doc_id in idx_dict[word]:
    #         lst.append((doc_id, idx_dict[word][doc_id]))
    #     idx_dict[word] = lst
    # #return idx_dict   
    for book_idx,book in enumerate(tokenized_books_feats):
        token = book[feature]
        term_count = Counter(token)
        for term, count in term_count.items():
            if term in idx_dict:
                idx_dict[term].append((book_idx, count))
            else:
                idx_dict[term] = [(book_idx, count)]
    for term in idx_dict:
        idx_dict[term].sort(key=lambda x: x[0])
    return idx_dict     


def build_inverted_indexes(tokenized_db_feats):
    print("STARTED BUILDING INV INDEXES")
    descript_idx = {}
    categories_idx = {}
    # go thru each book (dict) in the list
    # output 2 inverted indexes (one for each feat)
    descript_idx = build_idx_helper(descript_idx, tokenized_db_feats, "descript")
    categories_idx = build_idx_helper(categories_idx, tokenized_db_feats, "categories")
    print("FINISHED BUILDING INV INDEXES")
    return descript_idx, categories_idx


def compute_idf(inv_idx, n_docs, min_df=5, max_df_ratio=0.95):
    idf = {}
    for term, doc_ids in inv_idx.items():
        df = len(doc_ids)
        if df >= min_df and df / n_docs <= max_df_ratio:
            idf[term] = math.log2(n_docs / (1+df))
    return idf


def compute_doc_norms(index, idf, n_docs):
    norms = np.zeros(n_docs)

    for term, postings in index.items():
        idf_value = idf.get(term, 0)

        for doc_id, tf in postings:
            norms[doc_id] += (tf * idf_value) ** 2

    norms = np.sqrt(norms)
    return norms


def accumulate_dot_scores(query_word_counts, index, idf):
    doc_scores = {}
    for term, tf_query in query_word_counts.items():
        if tf_query != 0:
            idf_i = idf.get(term, 0)
            postings = index.get(term, [])
            for doc_id, tf_doc in postings:
                score = (tf_doc) * (idf_i) ** 2 * tf_query
                if doc_id in doc_scores:
                    doc_scores[doc_id] += score
                else:
                    doc_scores[doc_id] = score
    return doc_scores


def index_search(query_word_counts, doc_scores, idf, doc_norms):
    q_norm = 0
    for token, tf_freq in query_word_counts.items():
        if token in idf:
            num = (tf_freq * idf[token])**2
            q_norm += num
    q_norm = np.sqrt(q_norm)
    results = []
    for doc_id, score in doc_scores.items():
        denominator = (q_norm * doc_norms[doc_id])
        normalized_score = (score / denominator) 
        results.append((normalized_score, doc_id))

    results.sort(reverse=True)
    return results



def get_responses_from_results(response, results):
    print("STARTED GETTING RESPONSES")
    # Take results of index search and get list of books
    acc = []
    # print(results)
    for x in results:
        id = x[1]
        acc.append(response[id])
    return acc[:21]
