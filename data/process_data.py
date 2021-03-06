# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    load files and merge
    
        Parameter:
             messages_filepath (str) : message.csv
             categories_filepath(str) : categories.csv
             
        Return:
             df(DataFrame): merged file
             
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on=['id']) 
    
    return df


def clean_data(df):
    
    """
    Clean df
    
    Expand categories columns and get dummies for each category
    
    Drop rows with related == 2 as no information can be extracted based on the value
    
    Drop duplicated rows.
    
        Parameter:
            df(DataFrame): merged df
        
        Return:
            df(DataFrame): cleaned df
    """
    
    categories = df.categories.str.split(';', expand=True) # create a dataframe of the 36 individual category columns
    
    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df.drop(columns = ['categories'], axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], join='inner' ,axis=1)
    
    # remove rows related==2
    df = df[df['related']!=2]
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    
    """ 
    Save cleaned DataFrame to a SQLite database. 
  
    Parameters: 
        df (DataFrame):  cleaned dateframe
        database_filename (str) : database filename
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')
     


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