import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """This function loads both csv files and merges them on ID column.
        INPUT:
             messages_filepath (str) - path to messages data
             categories_filepath (str) - path to categories data
        OUTPUT:
             df (pd.DataFrame) - loaded, joined data"""
    messages = pd.read_csv(messages_filepath, encoding="utf-8)
    categories = pd.read_csv(categories_filepath, encoding="utf-8)
    df = messages.merge(categories, how="inner", on="id")
    return df


def clean_data(df):
    """This function cleans data, removes duplicates and extends categories by adding "not_related" category
    INPUT: 
        df (pd.DataFrame) - data containing messages with categories
    OUTPUT:
        df (pd.DataFrame) - preprocessed data"""
    categories = df.categories.str.split(';', expand = True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist()
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    categories.related.loc[categories.related == 2] = 0
      
    def create_not_related(x):
        if x == 0:
            return 1 
        else: 
            return 0
    
    categories["not_related"] = categories["related"].apply(create_not_related)
    df.drop('categories', axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)
    
    df.drop_duplicates(subset = 'id', inplace = True)
    
    return df


def augument_data(df):
    """As many labels seem to be given incorrectly (messages contain category keywords but aren't labeled as them)
    , when label keyword occurs in text it should be relevant for the label
    INPUT: 
        df (pd.DataFrame) - cleaned data with messages and categories
    OUTPUT:
        df (pd.DataFrame) - augumented data """
    
    category_names = list(df.columns[5:-1])
    for category in category_names:
        category_str= category.replace("_", " ")
        #numberofcases = len(df[(df["message"].str.contains(category_str)) & (df[category]==0)])
        print(len(df[(df["message"].str.contains(category_str)) & (df[category]==0)]), "changes made to column", category)
        df.loc[(df["message"].str.contains(category_str)) & (df[category]==0), category] = 1

        print(len(df.loc[((df[category]==1) & (df["related"]==0) ), "related"] ), "updates to column 'related'")
        df.loc[((df[category]==1) & (df["related"]==0) ), "related"] = 1
        
        df.loc[((df["related"]==1) & (df["not_related"]==1) ), "not_related"] = 0
                
    return df
    
        


def save_data(df, database_filename):
    """Function saves a Pandas DataFrame as a SQLite database.
        INPUT: 
            df (pd.DataFrame) - preprocessed data containing messages and categories
            database_filename (str) - name of the database
        OUTPUT: NONE"""
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql("Disasters", engine, index=False)
  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Preprocessing data...')
        df = clean_data(df)
        
        print("Augumenting data...")
        df = augument_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Preprocessed data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
