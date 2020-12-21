# import libraries 

from flask import Flask, request, jsonify, json
import requests
import pickle
import os
from tensorflow import keras
from keras.models import Sequential
from keras.preprocessing import text, sequence
from keras import utils
from keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.stem import SnowballStemmer 
from nltk.corpus import stopwords
import spacy
import pandas as pd



app = Flask(__name__)

# Prepare Elastic Search 

def load_keras_model(keras_file_path):

    """read the model form the path passed in arguments
    Args:
        Keras file path
    Returns:
        Keras Deep Learning Model 
    """

    # load keras deep learning model 
    keras_model = keras.models.load_model(keras_file_path)

    # return the model
    return keras_model




REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_\']')
STOPWORDS = set(stopwords.words('english'))

stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')

ps = nltk.PorterStemmer()
def stemming(text_sentence): 

    """"
    stemming function  reducing the word to its core root
    e.g. the words ending with “ed”, or “ing”
    """

    text = [ps.stem(word) for word in text_sentence.split()]
    return ' '.join(text)

wm = nltk.WordNetLemmatizer()
def lemmatize(text_sentence):

    """"
    Lemmatization is closely related to stemming. 
    It goes a steps further by linking words with similar meaning to one word.
    e.g. better -> good / was -> be 
    """

    text = [wm.lemmatize(word) for word in text_sentence.split()]
    return ' '.join(text)

def clean_text(text):

    """"
    Cleaning the text through remove the numbers, symbols and convert to lower case 
    """
    text = re.sub(r'\w*\d\w*', '', text).strip() # removes all words that contains numbers
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = re.sub('\d+', '', text)
    return text

def links_email_remover(text):

    """
    Remove the links in the text, 
    including the link that does not have a protocol like facebook.com    
    """
    text = re.sub(r"http\S+", "", text) # remove links with protocol
    text = re.sub(r"\w+[.]\S+|\w+[@]", "", text) # remove links without protocol
    return text

def stopwords_remover(text):

    """
    Remove the stop words from the text 
    e.g. the,of,on,with, etc..
    """
    text = re.sub('\'\w+', '', text) # Remove ticks and the next character 
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    text = ' '.join(word for word in text.split() if len(word) > 3) # remove stopwors from text 
    return text


def preprocess_text(text):
   text = links_email_remover(text)
   text = clean_text(text)
   text = lemmatize(text)
   text = stopwords_remover(text)
   text = stemming(text)
   return text



# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 500

def fromTextToFeatures(seq_text):

    """building features from text data
    Args:
        text (string): the text contains app name and description
    Returns:
        array: 
    """
    seq_text = [seq_text]
    # loading
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)    

    # gives you a list of integer sequences encoding the words in your sentence
    # seq_text : array of lists, each list represents the tokens of its text 
    X = tokenizer.texts_to_sequences(seq_text)
    # split the X 1-dimensional sequence of word indexes into a 2-d listof items
    # Each item is split is a sequence of 50 value left-padded with zeros
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    return X



def prepare_text(text):

    """ prepare the text sequence to the model
    Args:
        text (string): the text contains app name and description
    Returns:
        array: expected rating for the app 
    """

    text = preprocess_text(text)
    text_encoded = fromTextToFeatures(text)
    return text_encoded


def predict_rating(text):
 
    """predict app rating using the keras model 
    Args:
        text (string): the text contains app name and description
    Returns:
        int: expected rating for the app 
    """

    # prepare the text to the model 
    X = prepare_text(text)
    rating = keras_model.predict(X)
    return rating
        


"""
Flask App Routes
"""
@app.route("/")
def welcome():
    return "Welcome"



@app.route("/Predict_App_Rating", methods=['GET', 'POST'])

def get_rating():

    # if the method is POST
    if request.method == 'POST':

        # get the data from the POST request
        requested_data = request.get_json()

        # storing the arguments values in variables
        app_name = requested_data['app_name']
        app_description = requested_data['app_description']

        # merge the app name and description for prepare the text to the model
        text = app_name + app_description

        # call the predicting function and return the result
        return json.dumps(predict_rating(keras_model, text))

    # if the method is GET
    else:

        # get the text from the get request
        app_name = request.args.get('app_name')
        app_description = request.args.get('app_description')

        # merge the app name and description for prepare the text to the model
        text = app_name + app_description

        # call the predicting function and return the result
        return str(predict_rating(text)[0][3])






#----------------------------------------------------------------------------





@app.route("/Get_Relative_Apps_Rating", methods=['GET', 'POST'])

def get_rating_for_relative_apps():

    # if the method is POST
    if request.method == 'POST':

        # get the data from the POST request
        requested_data = request.get_json()

        # storing the arguments values in variables
        apps_type = requested_data['apps_type']

        # call the predicting function and return the result
        return json.dumps(predict_rating(keras_model, text))

    # if the method is GET
    else:

        # get the data from the get request
        apps_type = request.args.get('apps_type')


    """
     retrive relative apps to predict it's rating
    """

    # call the predicting function and return the result
    #return json.dumps(predict_rating(keras_model, text))
    return json.dumps(prepare_text(text))




"""
Flask App Entry Point (Main)
"""

   
if __name__ == '__main__':


    # get the model file path from the console
    #model_path = get_model_path()

    #if not os.path.isfile(model_path):
       # model file does not exist or not readable
     #  print("Error. Please make sure that the file is accessible")
      # print()

    #else :
        # read keras model
        keras_model = load_keras_model('FiveClassesModel.h5') #model_path
        app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))
        


