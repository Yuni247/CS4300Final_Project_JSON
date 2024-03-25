import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
#from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
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
@app.route("/simsuggestions")
def books_search():
    input_book_title = request.args.get("input_book")  # Get the book title from query parameters

    # Process the input book title to get the desired row in books_df
    book_row = books_df[books_df['Title'] == input_book_title].iloc[0]

    tokenized_books_feats = process_books_df(books_df)

    # Build inverted indexes for each feature
    authors_idx, descript_idx, categories_idx = build_inverted_indexes(tokenized_books_feats)

    num_rows = len(books_df)

    # Calculate idfs for each feature
    authors_idf = compute_idf(authors_idx, num_rows, min_df=5, max_df_ratio=0.95)
    descript_idf = compute_idf(descript_idx, num_rows, min_df=5, max_df_ratio=0.95)
    categories_idf = compute_idf(categories_idx, num_rows, min_df=5, max_df_ratio=0.95)

    # calculate doc norms for each feat
    authors_d_norms = compute_doc_norms(authors_idx, authors_idf, num_rows)
    descript_d_norms = compute_doc_norms(descript_idx, descript_idf, num_rows)
    categories_d_norms = compute_doc_norms(categories_idx, categories_idf, num_rows)

    def input_book_words(input_book, feature):
        words = {}
        for word in tokenize(input_book[feature]):
            if word in words:
                words[word] += 1
            else:
                words[word] = 1
    
    authors_inpbook_words, descript_inpbook_words, categories_inpbook_words = input_book_words(book_row, "authors"), input_book_words(book_row, "descript"), input_book_words(book_row, "categories")

    authors_scores = accumulate_dot_scores(authors_inpbook_words, authors_idx, authors_idf)
    descript_scores = accumulate_dot_scores(descript_inpbook_words, descript_idx, descript_idf)
    categories_scores = accumulate_dot_scores(categories_inpbook_words, categories_idx, categories_idf)

    # Top 20 results for each cossim comparison (authors, descript, and categories)
    authors_list = index_search(authors_inpbook_words, authors_scores, authors_idf, authors_d_norms)[0:30]
    descript_list = index_search(descript_inpbook_words, descript_scores, descript_idf, descript_d_norms)[0:30]
    categories_list = index_search(categories_inpbook_words, categories_scores, categories_idf, categories_d_norms)[0:30]

    def id_to_titles(input_list):
        output_list = []
        for book in input_list:
            title = books_df.iloc[book[1]]['Title']
            output_list.append((output_list[0], title))
        return output_list
    
    authors_list, descript_list, categories_list = id_to_titles(authors_list), id_to_titles(descript_list), id_to_titles(categories_list)
    
    # Calculate average scores
    avg_authors_score = sum(score for score, _ in authors_list) / len(authors_list)
    avg_descript_score = sum(score for score, _ in descript_list) / len(descript_list)
    avg_categories_score = sum(score for score, _ in categories_list) / len(categories_list)

    # creating final list based on common books between the three lists. Weights: authors > categories > descript
    combined_scores = {}
    for score, title in authors_list:
        combined_scores[title] = (score / avg_authors_score) * 5
    for score, title in descript_list:
        combined_scores[title] += (score / avg_descript_score) * 4
    for score, title in categories_list:
        combined_scores[title] += (score / avg_categories_score)

    # Convert the dictionary to a sorted list of tuples
    combined_list = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    ranked_results = combined_list.head(10)
    # for now round it to nearest 2 decimal places for uniformity, can change later
    ranked_results.loc[:, 'review_score'] = ranked_results['review_score'].round(2)

    top_recs = ranked_results[[
        'Title', 'descript', 'review_score']]

    return top_recs.to_json(orient='records')



if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5217)
