# import libraries
from flask import Flask, request, jsonify, json
import requests
import os
import tensorflow
import nltk
import re
import pickle
import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch
from nltk.corpus import stopwords
import os

app = Flask(__name__)

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


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_\']')
def clean_text(text):
    """"
    Cleaning the text through remove the numbers, symbols and convert to lower case 
    """
    text = re.sub(r'\w*\d\w*', '',
                  text).strip()  # removes all words that contains numbers
    text = text.lower()  # lowercase text
    # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing.
    text = BAD_SYMBOLS_RE.sub('', text)
    text = re.sub('\d+', '', text)
    return text


def links_email_remover(text):
    """
    Remove the links in the text, 
    including the link that does not have a protocol like facebook.com    
    """
    text = re.sub(r"http\S+", "", text)  # remove links with protocol
    # remove links without protocol
    text = re.sub(r"\w+[.]\S+|\w+[@]", "", text)
    return text


STOPWORDS = set(stopwords.words('english'))
def stopwords_remover(text):
    """
    Remove the stop words from the text 
    e.g. the,of,on,with, etc..
    """
    text = re.sub('\'\w+', '', text)  # Remove ticks and the next character
    # remove stopwors from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    text = ' '.join(word for word in text.split() if len(word)
                    > 3)  # remove stopwors from text
    return text


def preprocess_text(text):
    text = links_email_remover(text)
    text = clean_text(text)
    text = lemmatize(text)
    text = stopwords_remover(text)
    text = stemming(text)
    return text


def fromTextToFeatures(seq_text):
    """building features from text data
    Args:
        text (string): the text contains app name and description
    Returns:
        array: 
    """
    # gives you a list of integer sequences encoding the words in your sentence
    # seq_text : array of lists, each list represents the tokens of its text
    X = tokenizer.texts_to_sequences(seq_text)
    # split the X 1-dimensional sequence of word indexes into a 2-d listof items
    # Each item is split is a sequence of 50 value left-padded with zeros
    X = tensorflow.keras.preprocessing.sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    return X


def prepare_text(text):
    """ prepare the text sequence to the model
    Args:
        text (string): the text contains app name and description
    Returns:
        array: expected rating for the app 
    """
    text = list(map(lambda x: preprocess_text(x), text))
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
    return "Source code and documentation: https://github.com/You-Predict/Back-End"


@app.route("/predict", methods=['GET', 'POST'])
def predict_one_app():
    # if the method is POST
    if request.method == 'POST':
        # get the data from the POST request
        requested_data = request.get_json()
        if len(requested_data) < 2:
            return "bad request. you need to give the app name and description"
        # storing the arguments values in variables
        app_name = requested_data['app_name']
        app_description = requested_data['app_description']

    # if the method is GET
    elif request.method == 'GET':
        requested_data = request.args
        if len(requested_data) < 2:
            return "bad request. you need to give the app name and description"
        # get the app name from GET request args
        app_name = requested_data.get('app_name')
        # get the app description from GET request args
        app_description = requested_data.get('app_description')
    # merge the app name and description for prepare the text to the model
    text = app_name + " " + app_description
    # call the predicting function and return the result
    prediction_result = predict_rating([text])
    prediction_result = prediction_result[0].tolist()
    request_response = {
        "predicted_rating": prediction_result.index(max(prediction_result))+1}
    return jsonify(request_response)


@app.route("/search", methods=['GET', 'POST'])
def get_rating_for_relative_apps():
    """
     retrive relative apps to predict it's rating
    """
    # if the method is POST
    if request.method == 'POST':
        # get the data from the POST request
        requested_data = request.get_json()
        if len(requested_data) < 1:
            return "bad request. you need to give the app name"
        # storing the arguments values in variables
        app_name = requested_data['app_name']
        # call the predicting function and return the result

    # if the method is GET
    elif request.method == 'GET':
        requested_data = request.args
        if len(requested_data) < 1:
            return "bad request. you need to give the app name"
        # get the data from the get request
        app_name = requested_data.get('app_name')

    query = app_name
    # a query over the title and description
    query_seetings = {
        "query": {
            "multi_match": {
                "query": query
            }
        }
    }
    # using the search module to make the queryﻻ
    res = es.search(index=index_name, body=query_seetings)
    # response list
    response_list = res['hits']['hits']
    # if no app was found
    if len(response_list) == 0:
        return jsonify([])
    # adding neccessary keys to the dicts
    for item in response_list:
        item.update({"title": item['_source']['title'], "description": item['_source']
                     ['description'], "store_rating": item['_source']['store_rating']})
    df = pd.DataFrame(response_list)
    # removing unnecessary columns from the dataframe
    df = df.drop(columns=['_index', '_type', '_score', '_source'])
    # combining the name and description
    texts_for_preproccessing = [df['title']+' '+df['description']]
    texts_for_preproccessing = np.array(texts_for_preproccessing[0].values)
    # calling the prediction function
    prediction_result = predict_rating(texts_for_preproccessing)
    prediction_result = prediction_result.tolist()
    prediction_result = list(
        map(lambda x: x.index(max(x))+1, prediction_result))
    # create a column for the predictions
    df['prediction_rating'] = prediction_result
    # call the predicting function and return the result
    # return json.dumps(predict_rating(keras_model, text))
    df_json = df.to_json(orient="records")
    request_response = df_json
    return request_response


"""
Flask App Entry Point (Main)
"""
if __name__ == '__main__':
    # get the model file path from the console
    #model_path = get_model_path()

    # if not os.path.isfile(model_path):
    # model file does not exist or not readable
    #  print("Error. Please make sure that the file is accessible")
    # print()

    # else :
    # read keras model
    keras_model = tensorflow.keras.models.load_model('FiveClassesModel.h5')
    # loading keras tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 500
    # connecting to elasticsearch endpoint
    es = Elasticsearch("https://0f09431a2786444ab99c6826be973c68.europe-west3.gcp.cloud.es.io:9243",
                       http_auth=('elastic', 'zIXNAI2UfZVPTqVxd3LllsGe'))
    # name of the created index
    index_name = "applications"
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=port)
