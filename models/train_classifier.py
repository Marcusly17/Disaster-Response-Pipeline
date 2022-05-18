import sys

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag

from sklearn.metrics import confusion_matrix , classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

from sqlalchemy import create_engine

import pickle


def load_data(database_filepath):
    """ 
    Loading data 
  
    Return X, y, and categories name
  
        Parameters:
            database_filepath (str): database filepath 

        Returns:
            X (DataFrame): feature columns
            y (DataFrame): label columns
            category_names (list): category names
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)
    
    X = df.message
    y = df.iloc[:,4:]
    
    return X, y, y.columns.tolist()

def tokenize(text):
    """ 
    Normalize, tokenize and lemmatize function. 
  
        Parameters: 
            text (str): Text for cleaning.
    
        Returns: 
            clean_tokens (list): cleaned tokens for ML training.
    """
    #nomalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    stop_words = stopwords.words("english")
    #tokenize
    tokens = word_tokenize(text)
    #lemmatize
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens

#build customized feature
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    A class to check whether the first word of the sentence is verb,
    adding a new feature for the ML classifier
    """

    def starting_verb(self, text):
        """
        return 1 if first word is verb, otherwise 0
        
        """
        sentence_list = sent_tokenize(text)
        
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(word_tokenize(sentence))
            
            first_word, first_tag = pos_tags[0]
            
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return 1
            
        return 0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    """
    Build Model funcion
    
    Create pipeline and use GridSearch
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ])),

                ('starting_verb', StartingVerbExtractor())
            ])),


        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    # parameters to grid search
    parameters = {
        'clf__estimator__n_estimators': [50, 80, 100]
    }

    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model
    
    Use fit model to predict y 
    
    Print f1 score, precision, recall and accuracy for each categories
    
        Parameters:
            model (GridSearchCV) : trained model
            X_test (DataFrame) : test features
            Y_test (DataFrame) : test label
            category_names (list): category name
            
    """
    y_pred = model.predict(X_test)
    
    #print model evaluvation result by categories
    for i, column in enumerate(category_names):
        print(column, "\n", classification_report(Y_test.values[:,i], y_pred[:,i]), "\n", \
              column, " accuracy_score: ", accuracy_score(Y_test.values[:,i], y_pred[:,i]),"\n",\
              "-"*65,"\n")

def save_model(model, model_filepath):
    """
    
    Save trained model as a pickle which can be loaded later
    
    """
        pickle.dump(model, open(model_filepath,'wb'))
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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