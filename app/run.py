import re
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar, Heatmap
from sqlalchemy import create_engine
import joblib



app = Flask(__name__)

def tokenize(text):
    """tokenize function processes message text data - lowercase, no punctuation, lemmatization, URLs
    INPUT: 
           text (str)- message text
    OUTPUT: 
           cleaned_tokens (list) - List of tokens"""
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls_found = re.findall(url_regex, text)
    for url in urls_found:
        text = text.replace(url, 'URL')
    email_regex = '\w+@\w+\.\w+'
    emails_found = re.findall(email_regex, text)
    for email in emails_found:
        text = text.replace(email, 'EMAIL')
    text = re.sub(r"[^a-zA-Z0-9]"," ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        cleaned_tokens.append(clean_tok)
    return cleaned_tokens

# load database previously saved in data/ folder
engine = create_engine("sqlite:///data/DisasterData.db")
#df = pd.read_sql_table("Disasters", engine)
df = pd.read_sql("SELECT * FROM Disasters", engine)

# load model previously saved in models/ folder.
model = joblib.load("models/classifier_cv.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    
    message_counts = df.drop(['id','message','original','genre'], axis=1).sum().sort_values()
    message_names = list(message_counts.index)
    
    #correlations
    category_correlations = df.iloc[:,4:].corr().values
    category_names = list(df.iloc[:,5:-1].columns)

  

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
      {
            'data': [
                Bar(
                    x=message_counts,
                    y=message_names,
                    orientation = 'h',
                    
                
                )
            ],
           
            'layout': {
                'title': 'Distribution of messages per category',
                'yaxis': {
                        'title': "Category"
                        },
                
                'xaxis': {
                    'title': "Number of Messages"
                    },
                    
            }
        },
                
         {
            'data': [
                Heatmap(
                    x=category_names,
                    y=category_names,
                    z=category_correlations
                )    
            ],

            'layout': {
                'title': 'Correlations of labels'
            }
        },
                
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
