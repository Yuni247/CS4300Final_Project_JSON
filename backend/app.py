import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
import Levenshtein as lev

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

app = Flask(__name__)
CORS(app)

# Sample search using json with pandas


def json_search(query):
    books_df['Edit_distance'] = books_df['Title'].apply(
        lambda book: lev.distance(query.lower(), book.lower()))

    sorted_df = books_df.sort_values(by='Edit_distance', ascending=True)

    ranked_results = sorted_df.head(10)
    # for now round it to nearest 2 decimal places for uniformity, can change later
    ranked_results['review_score'] = ranked_results['review_score'].round(2)

    top_recs = ranked_results[[
        'Title', 'descript', 'review_score']]

    return top_recs.to_json(orient='records')


@app.route("/")
def home():
    return render_template('base.html', title="sample html")


@app.route("/books")
def episodes_search():
    text = request.args.get("title")
    return json_search(text)


if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)


"""
import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from nltk.tokenize import TreebankWordTokenizer
from cossim import * 


# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
MYSQL_USER = "root"
MYSQL_USER_PASSWORD = "root"
MYSQL_PORT = 3306
MYSQL_DATABASE = "cs4300_booksdb"

mysql_engine = MySQLDatabaseHandler(MYSQL_USER,MYSQL_USER_PASSWORD,MYSQL_PORT,MYSQL_DATABASE)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

# Sample search, the LIKE operator in this case is hard-coded, 
# but if you decide to use SQLAlchemy ORM framework, 
# there's a much better and cleaner way to do this
# Dictionary ={1:'Welcome', 2:'to',
#             3:'Geeks', 4:'for',
#             5:'Geeks'}
def sql_search(query):
    query_sql = """SELECT Title, descript, authors, publisher, categories, review_score, review_count FROM new_books_merged"""
    data = pd.read_sql(query_sql, mysql_engine)
    keys = ["Title", "descript", "authors", "publisher", "categories", "review_score", "review_count"]
    return json.dumps(data.to_dict(orient='records'))

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/books")
def books_search():
    query = request.args.get("title")
    response = json.loads(sql_search(query))
    r_score_dict = {}
    r_count_dict = {}
    response_arr = []

    for i in range(len(response)):
        result = response[i]
        desc = result['descript']
       
        descript_toks = TreebankWordTokenizer().tokenize(desc)
        result['toks'] = descript_toks
        r_score_dict[i] = result['review_score']
        r_count_dict[i] = result['review_count']
        response_arr.append(result)
    
    inv_idx = build_inverted_indexes(response)
 
    idf = compute_idf(inv_idx=inv_idx, n_docs=len(response))

    doc_norms = compute_doc_norms(inv_idx, idf, len(response))
    query_words = {}
    for word in TreebankWordTokenizer().tokenize(query):
        if word in query_words:
            query_words[word] += 1
        else:
            query_words[word] = 1

    inv_idx = {key: val for key, val in inv_idx.items() if key in idf}

    scores = accumulate_dot_scores(query_words, inv_idx, idf)
    results = index_search(query, inv_idx, idf, doc_norms, scores, r_score_dict, r_count_dict)
    for i in results:
        score = i[0]
        id = i[1]
        
        response[id]['cosine'] = score
    user_results = get_responses_from_results(response, results)
    return user_results
"""
