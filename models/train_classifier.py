import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, classification_report
import joblib

#probably an IDE-issue, loading functions from another file from the project
try:
    from custom_transformers import * 
except ModuleNotFoundError:
    from .custom_transformers import *



def load_data(database_filepath):
    """Function loads data previously saved in a database
    INPUT: 
        database_filepath (str) - path to SQLite database
    OUT: 
        X (pd.Series)- Series object of message data
        Y (pd.DataFrame) - DataFrame of categories (columns to be predicted)
        """
    
    sql_engine = create_engine(
        'sqlite:///' + str(database_filepath), echo=False)
    table_name = str(sql_engine.table_names()[0])
    print('DB table names', sql_engine.table_names())
    df = pd.read_sql_table(table_name, con=sql_engine)

    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """tokenize function processes message text data - lowercase, no punctuation, lemmatization, URLs
    INPUT: 
        text (str) - text to tokenize
    OUTPUT: 
        cleaned_tokens (list) - list of tokens"""
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


def build_model(grid_search=False):
    """Constructs the machine learning Pipeline.
    INPUT: 
        grid_search (boolean): if true, a grid search is performed to find the most optimal model
    OUTPUT:
        Sklearn Pipeline (with or without GridSearch) """
    pipeline = Pipeline(
    [("ft", FeatureUnion([
        ("txt", Pipeline([('vect', CountVectorizer(tokenizer = tokenize, ngram_range =(1,2))), 
                     ('tfidf', TfidfTransformer())]) )
        ,("mess_cnt_pers_pron", MessagePersonalPronounsUsed())
        ,("mess_cnt_poss_pron",MessagePossesivePronounsUsed())
        ,("mess_cnt_prop_nouns", MessageProperNounsUsed())
        ,("mess_cnt_common_nouns", MessageNounsUsed())
        ,("mess_length", MessageLengthExtractor())
        ,("mess_avg_word", MessageAvgWordExtractor())
        ,("mess_cnt_numerics", MessageNumericsExtractor())
        ,("mess_cnt_uppercase", MessageUppercaseExtractor())
        ,("mess_cnt_urls_and_emails", UrlsEmailsExtractor())
          ]))
    
        , ('clf', MultiOutputClassifier(RandomForestClassifier()))
              ])

    scorer = make_scorer(f1_score,average='micro')

    params ={"clf__estimator__n_estimators" : [100, 500, 1000]
        ,"clf__estimator__min_samples_leaf" : [2,3,5,7]}
    
    if grid_search == True:
        cv = GridSearchCV(pipeline, param_grid = params, scoring= scorer, cv = 3) 
        return cv
    
    return pipeline


def evaluate_model(model, X_test, Y_test, model_filepath, category_names):
    """Evaluate the model for each category and save results as csv file.
    Input:
        model (sklearn Pipeline) - name of model
        X_test (pd.DataFrame) -  Column with messages
        Y_test (pd.DataFrame) - Encoded columns of categories
        model_filepath (str) - path of model (used to save results under same name as Pickle file)
        category_names (list) - list of names of target categories (labels)
    Output:
        None
    """
    Y_pred = model.predict(X_test.values)    
    #output_dict from https://stackoverflow.com/a/53780589
    report_dict = classification_report(Y_test, Y_pred, target_names = category_names, output_dict=True)
    evaluation = pd.DataFrame(report_dict)
    results_filepath = model_filepath.split('.')[0] + ".csv"
    evaluation.transpose().to_csv(results_filepath)

def save_model(model, model_filepath):
    """Save model as a pickle file
    Input:
      model (sklearn Pipeline): name of model
      model_filepath (str): specified path
    Output: None
    """
    joblib.dump(model, str(model_filepath) ,  compress=4)
      

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
        
        print('Building model...')
        model = build_model(grid_search=True)
        
        print('Training model, it can take a while...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, model_filepath, category_names)
        
        print('Evaluation results saved under {}'.format(model_filepath.split('.')[0])+ '.csv')
       
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        



    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()


    
