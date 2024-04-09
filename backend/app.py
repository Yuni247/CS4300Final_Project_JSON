import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import Levenshtein as lev
from cossim import * 
from cossim import build_inverted_indexes

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

app = Flask(__name__, static_folder='static')
CORS(app)

# FOR COSSIM: Processing DB, creating inverted indexes here

# tokenized_books_feats --> list of dicts of tokenized ["authors"], ["descript"], ["categories"] strings (each book has a dict)
tokenized_books_feats = process_books_df(books_df)

# Build inverted indexes for each feature
descript_idx, categories_idx = build_inverted_indexes(tokenized_books_feats)

num_rows = len(books_df)

# Calculate idfs for each feature
descript_idf = compute_idf(descript_idx, num_rows, min_df=5, max_df_ratio=0.95)
categories_idf = compute_idf(categories_idx, num_rows, min_df=5, max_df_ratio=0.95)

# calculate doc norms for each feature
descript_d_norms = compute_doc_norms(descript_idx, descript_idf, num_rows)
categories_d_norms = compute_doc_norms(categories_idx, categories_idf, num_rows)

# FOR COSSIM: End of pre-processing




# # Sample search using json with pandas
# def get_books_titles():

#     titles = []
#     for bk in data:
#         titles.append(bk["Title"])

#     with open("books_titles", "w") as jsonf:
#         jsonf.write(json.dumps(titles))

#print(get_books_titles())

@app.route("/suggest")
def title_search():
    query = request.args.get("title")
    if query != None:
        # print(query.lower())
        books_df['Edit_distance'] = books_df['Title'].apply(
            lambda book: lev.distance(query.lower(), book.lower()))

        sorted_df = books_df.sort_values(by='Edit_distance', ascending=True)

        ranked_results = sorted_df.head(10)
        # for now round it to nearest 2 decimal places for uniformity, can change later
        ranked_results.loc[:, 'review_score'] = ranked_results['review_score'].round(2)

        top_recs = ranked_results[[
            'Title', 'descript', 'review_score']]

        return top_recs.to_json(orient='records')
    return []


@app.route("/")
def home():
    return render_template('base.html', title="sample html")


#input- input_book: row in books_df that was matched with the user's query input
@app.route("/books")
def books_search():
    input_book_title = request.args.get("title")  # Get the book title from query parameters

    # Process the input book title to get the desired row in books_df
    book_row = books_df[books_df['Title'] == input_book_title].iloc[0]

    def input_book_words(input_book, feature):
        words = {}
        if feature == "descript":
            for word in tokenize(input_book["descript"] + input_book["authors"]):
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1
        elif feature == "categories":
            for word in tokenize(input_book[feature]):
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1
        return words
    
    descript_inpbook_words, categories_inpbook_words = input_book_words(book_row, "descript"), input_book_words(book_row, "categories")

    descript_scores = accumulate_dot_scores(descript_inpbook_words, descript_idx, descript_idf)
    categories_scores = accumulate_dot_scores(categories_inpbook_words, categories_idx, categories_idf)

    # Use Jaccard Similarity for authors (each author token is a single author, inside a list of toks for each book)
    authors_scores = []
    for id, book in enumerate(tokenized_books_feats):
        tok_df_authors = book["authors"]
        tok_inp_authors = tokenize_authors(book_row["authors"])
        intersection = list(set(tok_df_authors) & set(tok_inp_authors))
        union = list(set(tok_df_authors) | set(tok_inp_authors))
        score = len(intersection) / len(union)
        authors_scores.append((score, id))

    # Top 30 results for each cossim comparison (authors, descript, and categories)
    authors_list = sorted(authors_scores, key=lambda x: x[0], reverse=True)[0:30]
    descript_list = index_search(descript_inpbook_words, descript_scores, descript_idf, descript_d_norms)[0:30]
    categories_list = index_search(categories_inpbook_words, categories_scores, categories_idf, categories_d_norms)[0:30]

    def id_to_titles(input_list):
        output_list = []
        for book in input_list:
            title = books_df.iloc[book[1]]['Title']
            output_list.append((title, book[0]))
        return output_list
    
    authors_list, descript_list, categories_list = id_to_titles(authors_list), id_to_titles(descript_list), id_to_titles(categories_list)
    
    # Calculate average scores
    avg_authors_score = sum(score for _, score in authors_list) / len(authors_list)
    avg_descript_score = sum(score for _, score in descript_list) / len(descript_list)
    avg_categories_score = sum(score for _, score in categories_list) / len(categories_list)

    # creating final list based on common books between the three lists. Weights: authors > categories > descript
    combined_scores = {}

    # normal weights: (author, descript, categories: 0.6, 0.1, 0.3)
    # weight if authors = "null": (author, descript, categories: 0.0, 0.3, 0.7)
    if book_row["authors"] == "null" and book_row["descript"] != "null":
        for title, score in authors_list:
            combined_scores[title] = 0
        for title, score in descript_list:
            combined_scores[title] = combined_scores.get(title, 0) + (score / avg_descript_score) * 3
        for title, score in categories_list:
            combined_scores[title] = combined_scores.get(title, 0) + (score / avg_categories_score) * 7
    # weight if descript = "null": (author, descript, categories: 0.55, 0.0, 0.45)
    elif book_row["descript"] == "null" and book_row["authors"] != "null":
        for title, score in authors_list:
            combined_scores[title] = (score / avg_authors_score) * 5.5
        for title, score in descript_list:
            combined_scores[title] = 0
        for title, score in categories_list:
            combined_scores[title] = combined_scores.get(title, 0) + (score / avg_categories_score) * 4.5
    # weight if both authors and descript = "null": (author, descript, categories: 0.0, 0.0, 1.0)
    elif book_row["descript"] == "null" and book_row["descript"] == "null":
        for title, score in authors_list:
            combined_scores[title] = 0
        for title, score in descript_list:
            combined_scores[title] = 0
        for title, score in categories_list:
            combined_scores[title] = combined_scores.get(title, 0) + (score / avg_categories_score) * 10
    # Local change only (change this after p03 deadline), Bone by Bone author feature was wrongly overtaken by categories + descript. 
    else:
        for title, score in authors_list:
            combined_scores[title] = (score / avg_authors_score) * 4.5
        for title, score in descript_list:
            combined_scores[title] = combined_scores.get(title, 0) + (score / avg_descript_score) * 2
        for title, score in categories_list:
            combined_scores[title] = combined_scores.get(title, 0) + (score / avg_categories_score) * 3.5
    
    # Convert the dictionary to a sorted list of tuples
    combined_list = sorted(combined_scores.items(), key=lambda x: x[0], reverse=True)
    for result in combined_list:
        if result[0] == book_row["Title"]:
            combined_list.remove(result)

    combined_df = pd.DataFrame(combined_list, columns=[ 'Title', 'cossim_score'])
    combined_df['Title'], books_df['Title'] = combined_df['Title'].astype(str), books_df['Title'].astype(str)

    final_df = pd.merge(combined_df, books_df[['Title', 'authors', 'categories', 'descript', 'review_score', 'review_count']], on='Title', how='left')
    
    # sort the final df by the cossim scores, take only the top 10
    top_recs = final_df.sort_values(by='cossim_score', ascending=False).head(10)

    # review_score and review_count ranking system added
    top_recs['Weighted_Score'] = top_recs['review_score'] + np.log(top_recs['review_count'])
    top_recs_w_reviews = top_recs.sort_values(by='Weighted_Score', ascending=False)

    top_recs_w_reviews['cossim_score'], top_recs_w_reviews['Weighted_Score'] = top_recs_w_reviews['cossim_score'].round(2), top_recs_w_reviews['Weighted_Score'].round(2)

    top_recs_json = top_recs_w_reviews.to_json(orient='records')
    return top_recs_json


if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5217)
