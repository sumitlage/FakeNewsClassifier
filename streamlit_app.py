# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 13:11:14 2020

@author: slage
"""

from flask import Flask,request
import pandas as pd
import numpy as np


import pickle
import streamlit as st

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import warnings
warnings.filterwarnings('ignore')

pickel_in = open('FakeClassifier.pkl','rb')
classifier = pickle.load(pickel_in)

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

def predict(title):
    
    clean_title=text_analyzer(title)
    clean_title=get_embedded_corpus(5000,25,clean_title)
    
    result = classifier.predict([[clean_title]])
    
    return str(result)

def main():

    st.title("Fake CLassification Predictive Model")
    title = st.text_input("title","Type here")
    
    if st.button("Predict"):
        
       result =  predict(title)
       
       if(int(result)==0):
           st.success("Out is Real")
       else:
           st.success("Out is Fake")
       
    
if __name__ == '__main__':
    main()