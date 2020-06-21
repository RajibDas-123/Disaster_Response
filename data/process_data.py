import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    messages = pd.read_csv("messages.csv")
    categories = pd.read_csv("categories.csv")
    df = pd.merge(messages, categories, on="id")

    #exctracting all the caegories    
    categories = categories.categories.str.split(";", expand=True)
    row = categories.loc[0]
    
    category_colnames = row.apply(lambda cat: cat[:-2])
    
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda val: val[-1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    df = df.drop('categories', axis=1) 
    
    df = pd.concat([df, categories], axis=1)

    return df


def clean_data(df):
    
    df.iloc[:, 4:] = df.iloc[:,4:].applymap(lambda x: 1 if x > 1 else x)
    
    df = df.drop_duplicates(keep='first')
    
    df = df.dropna(subset=category_colnames, how='all')
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster_data', engine, index=False)
      


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()