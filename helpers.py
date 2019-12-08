import nltk
import sklearn
import glob
import pickle
import numpy as np
import copy
import random
import os
import pandas as pd
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn import model_selection

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from nltk import sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

stop_word_set = set(ENGLISH_STOP_WORDS)
chars = set(list('abcdefghijklmnopqrstuvwxyz0123456789'))

class LemmaTokenizer(object):
      def __init__(self):
            self.lemmatizer = WordNetLemmatizer()  
      def __call__(self, doc):  
            return [self.lemmatizer.lemmatize(t) for t in word_tokenize(doc)]  
        
class StemTokenizer(object):
      def __init__(self):
            self.stemmer = PorterStemmer()  
      def __call__(self, doc):  
            return [self.stemmer.stem(t) for t in word_tokenize(doc)]
        
class SnowballStemTokenizer(object):
      def __init__(self):
            self.stemmer = SnowballStemmer(language='english')  
      def __call__(self, doc):  
            return [self.stemmer.stem(t) for t in word_tokenize(doc)]

def misc_features(sents):
    features_list = []
    
    for sent in sents:
        features = {}
        
        features['upper_count'] = len(re.findall('[A-Z]', sent))
        features['lower_count'] = len(re.findall('[a-z]', sent))
        
        sent = sent.lower()
        features['word_count'] = len(sent.split())
        
        for c in chars:
            features[c] = sent.count(c)
            
        features_list.append(list(features.values()))

    return np.array(features_list)

def cleaner(sents):
    cleaned_sents = []

    for sent in sents:
        try:
          filt_sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
          filt_sent = ' '.join([w for w in filt_sent.split() if not (w.isalpha() and len(w)==1) and w.lower() not in stop_word_set] )
          cleaned_sents.append(filt_sent)
        except:
          print(sent)

    return cleaned_sents

def get_tfidf_vectorizer(x_train, max_df=.5, min_df = 2, stop_words='english',
                             lowercase=True, tokenizer=StemTokenizer(),
                             strip_accents='ascii', ngram_range=(1,1),
                            binary=False, analyzer='word'):
    
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, 
                                 stop_words=stop_words,
                             lowercase=lowercase, tokenizer=tokenizer,
                             strip_accents=strip_accents, 
                                 ngram_range=ngram_range, binary=binary, analyzer=analyzer)
    
    vectorizer.fit(x_train)
    
    return vectorizer