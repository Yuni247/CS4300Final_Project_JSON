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
