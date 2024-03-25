import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
#from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import Levenshtein as lev
from nltk.tokenize import TreebankWordTokenizer
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

# def json_search(query):
#     r_score_dict = {}
#     r_count_dict = {}
#     for doc_id,response in enumerate(responses):
#         desc = response['descript']
#         descript_toks = TreebankWordTokenizer().tokenize(desc)
#         response['toks'] = descript_toks
#         r_score_dict[doc_id] = response['review_score']
#         r_count_dict[doc_id] = response['review_count']
    

#     inv_idx = build_inverted_indexes(responses)
#     idf_author = compute_idf(inv_idx=inv_idx[2], n_docs=len(responses))
#     doc_norms = compute_doc_norms(inv_idx[2], idf_author, len(responses))

#     results = index_search(query, inv_idx[2], idf_author, doc_norms, r_score_dict, r_count_dict)
#     for i in results:
#         score = i[0]
#         id = i[1]
        
#         responses[id]['cosine'] = score
#     user_results = get_responses_from_results(responses, results)
#     #print(user_results)
#     return user_results

@app.route("/")
def home():
    return render_template('base.html', title="sample html")


@app.route("/books")
def books_search():
    query = request.args.get("title")
    responses = json.loads(json_search(query))
    return responses
    

if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5217)
