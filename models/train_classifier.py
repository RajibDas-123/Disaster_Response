import sys
import pandas as pd
import re
from sqlalchemy import create_engine
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from scipy.stats import hmean
from scipy.stats.mstats import gmean

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """    
    This class is responsible for extracting starting verb from the sentence.
    This starting verb will be used as a new feature.
    """

    def get_starting_verb(self, text):
        sent_lst = nltk.sent_tokenize(text)
        for sent in sent_lst:
            postags = nltk.pos_tag(tokenize(sent))
            try:
                fst_wrd, fst_tag = postags[0]
                if fst_tag in ['VB', 'VBP'] or fst_wrd == 'RT':
                    return True
            except:
                return False        
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new_features = pd.Series(X).apply(self.get_starting_verb)
        return pd.DataFrame(X_new_features)

def load_data(database_filepath):
    """
    Function to load the datasets
    
    Arguments:
        database_filepath -> path to SQLite db
    Output:
        X -> feature DataFrame
        Y -> label DataFrame
        category_names -> used for data visualization (app)
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * from disaster", engine)
    X = df.message.values
    y = df.iloc[:,4:].values
    categories = df.iloc[:,4:].columns.values
    return X, y, categories


def tokenize(text):
    """
    Function to get tokens from a text
    
    Arguments:
        text -> string to tokenize
    Output:
        clean_token -> list of tokens
    """

    url_pat = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    found_urls = re.findall(url_pat, text)
    for url in found_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_token = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens
        


def build_model():
    """
    Build Model function
    
    This function output is a Scikit ML Pipeline that process text messages
    according to NLP best-practice and apply a classifier.
    
    Arguments:
        None
    Output:
        None

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
    
    params = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.6, 1.0),
        'features__text_pipeline__vect__max_features': (None, 6000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
    }

    cv = GridSearchCV(pipeline, param_grid=params)
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function
    
    This function applies ML pipeline to a test set and prints out
    model performance (accuracy and f1score)
    
    Arguments:
        model -> Scikit ML Pipeline
        X_test -> test features
        Y_test -> test labels
        category_names -> label names (multi-output)
    Output:
        None

    """
    Y_pred = model.predict(X_test)
    accuracy = (Y_pred == Y_test).mean().mean()
    df_pred = pd.DataFrame(Y_pred, columns=category_names)
    df_test = pd.DataFrame(Y_test, columns=category_names)
    print('Average accuracy {0:.2f}% \n'.format(accuracy*100))

    for catg in category_names:
        print('------------------------------------------------------\n')
        print('FEATURE: {}\n'.format(catg))
        print(classification_report(df_test[catg],df_pred[catg]))

    


def save_model(model, model_filepath):  
    """
    Save Model function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        model -> GridSearchCV or Scikit Pipelin object
        model_filepath -> destination path to save .pkl file
    Output:
        None
    """    

    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)


def main():
    """
    Main function
    
    This function implements the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
    
    """
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
