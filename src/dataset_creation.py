import numpy as np
import pandas as pd
import os
# importing config file
import sys
import config,ner_f1
from nltk import word_tokenize
from nltk import pos_tag
import daiquiri,logging
daiquiri.setup(level=logging.INFO)
logger = daiquiri.getLogger()

#Loading data
logger.info("Loading raw data...")
train_df = pd.read_csv(config.RAW_DATA+"train.csv")
test_df = pd.read_csv(config.RAW_DATA+"test.csv")
sample_submission = pd.read_csv(config.RAW_DATA+"sample_submission.csv")

#Making the column Word as string
logger.info("Changing data type to string...")
train_df['Word'] = train_df['Word'].astype(str)
test_df['Word'] = test_df['Word'].astype(str)

#Get full sentence for each Sent_ID
class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: ' '.join(s['Word'].values.tolist())
        self.grouped = self.data.groupby(['Sent_ID']).apply(agg_func)
        self.sentences = [s for s in self.grouped]
        
    def get_next(self):
        try: 
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s 
        except:
            return None

# For each sentence get POS TAG for each word
def get_pos_tag(df):
    getter = SentenceGetter(df)
    sentences = getter.sentences
    
    docs = [sent.split(' ') for sent in sentences]
    final = [pos_tag(token) for token in docs]
    final_list = [(index+1,val) for (index,sublist) in enumerate(final) for val in sublist]
    final_list2= [(ind+1,x) for ind,x in enumerate(final_list)]
    final_df = pd.DataFrame([x for x in final_list2], columns=['id','tup'])
    final_df = final_df.join(pd.DataFrame(final_df.tup.tolist(), columns=['Sent_ID', 'tup2']))
    final_df.drop('tup',axis=1,inplace=True)
    final_df = final_df.join(pd.DataFrame(final_df.tup2.tolist(), columns=['Word', 'POS_TAG']))
    final_df.drop('tup2',axis=1,inplace=True)
    
    df_all = df.join(final_df[['POS_TAG']])
    return(df_all)
logger.info("Extracting POS_TAG for each Word in the training dataset")
train = get_pos_tag(train_df)
logger.info("Extracting POS_TAG for each Word in the testing dataset")
logger.info("Loading raw data...")
test = get_pos_tag(test_df)

#dumping processed data
logger.info("Dumping data...")
import pickle
pickle.dump(train, open( config.PROCESS_DATA+"train_features.p", "wb" ) )
pickle.dump(test, open( config.PROCESS_DATA+"test_features.p", "wb" ) )
logger.info("DONE!!!")