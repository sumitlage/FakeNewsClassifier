from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn

import tensorflow as tf
graph = tf.get_default_graph()

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import re

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot



app = Flask(__name__)
model = pickle.load(open('FakeClassifier.pkl', 'rb'))

def text_analyzer(messages):
    porter = PorterStemmer()
        
    text = messages
    clean_text = re.sub('[^a-zA-Z]',' ',text)
        
    clean_text = clean_text.lower()
    clean_text = clean_text.split()
        
    clean_text = [porter.stem(word) for word in clean_text if word not in stopwords.words('english')]
    clean_text = ' '.join(clean_text)

    return clean_text

def get_embedded_corpus(vocsize,sentlen,message):
    try:
        one_hot_corpus = [one_hot(word,vocsize) for word in message]
        padded_corpus = pad_sequences(one_hot_corpus,padding='pre',maxlen=sentlen)
    except e:
        print('exception is',e)
    finally:
        return padded_corpus


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():

    if request.method == 'POST':
        title = request.form['title']

        clean_title=text_analyzer(title)
        clean_title1=get_embedded_corpus(5000,50,clean_title)
    
        result = 0
        

        with graph.as_default():
            result = model.predict_classes(clean_title1)
    
        if result==0:
            return render_template('index.html',prediction_texts="Title is Real")
        else:
            return render_template('index.html',prediction_texts="Title is Fake")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
