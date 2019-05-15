import numpy as np
import pandas as pd
import os
import sys
import pickle
import config,ner_f1
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter
from itertools import chain

import daiquiri,logging
daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger()

logger.info("Loading Processed data...")
train_df = pickle.load( open( config.PROCESS_DATA+"train_features.p", "rb" ) )
test_df = pickle.load( open( config.PROCESS_DATA+"test_features.p", "rb" ) )
sample_submission = pd.read_csv(config.RAW_DATA+"sample_submission.csv")

logger.info("Changing data type...")
train_df['Word'] = train_df['Word'].astype(str)
test_df['Word'] = test_df['Word'].astype(str)
train_df['POS_TAG'] = train_df['POS_TAG'].astype(str)
test_df['POS_TAG'] = test_df['POS_TAG'].astype(str)


#Custom function to reshape training data into desired format
class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(), 
                                                           s['POS_TAG'].values.tolist(), 
                                                           s['tag'].values.tolist())]
        self.grouped = self.data.groupby(['Sent_ID']).apply(agg_func)
        self.sentences = [s for s in self.grouped]
        
    def get_next(self):
        try: 
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s 
        except:
            return None

def get_sentences(df):
    getter = SentenceGetter(df)
    sentences = getter.sentences
    return sentences


test_df['tag']=np.nan
train_sentences = get_sentences(train_df)
test_sentences = get_sentences(test_df)


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')

# Get features for each sentence in a dictionary format
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'lemma': stemmer.stem(word),
        'bias': 1.0, 
        'word.lower()': word.lower(), 
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit()
        #'postag': postag
        #'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            #'lemma1': stemmer.stem(word1),
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper()
            #'-1:postag': postag1
            #'-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            #'lemma1': stemmer.stem(word1),
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper()
            #'+1:postag': postag1
            #'+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
    return features
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent):
    return [label for token, postag, label in sent]
def sent2tokens(sent):
    return [token for token, postag, label in sent]

logger.info("Get features for training data...")
X = [sent2features(s) for s in train_sentences]

logger.info("Get target labels for training data...")
y = [sent2labels(s) for s in train_sentences]

logger.info("Get features for test data...")
X_predict = [sent2features(s) for s in test_sentences]

logger.info("Training CRF model from. SKLEARN pacakge...")
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X, y)

logger.info("Predicting the labels for test data...")
y_score = crf.predict(X_predict)
y_score_enc = list(chain.from_iterable(y_score))

logger.info("Preparing the submission file...")
sample_submission['tag'] = y_score_enc
sample_submission.to_csv(config.RESULTS+'final_submission.csv',index=False)

logger.info("DONE!!!")