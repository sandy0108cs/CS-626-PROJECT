# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 06:12:38 2020

@author: 91799
"""

import nltk
from nltk.corpus import brown as br
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM,Dropout, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score ,classification_report
from sklearn.metrics import confusion_matrix
from keras.preprocessing.sequence import pad_sequences
from keras import backend as acc
import math
def bi_lstm_model():    
    model = Sequential()
    model.add(InputLayer(input_shape=(MAX_LEN, )))
    model.add(Embedding(len(windex), 64))
    model.add(Bidirectional(LSTM(128, return_sequences=True,recurrent_dropout=0.2)))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(len(tindex),activation='softmax')))
#    model.add(Activation('sigmoid'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy', ignore_class_accuracy(0)])
    return model




 
def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = acc.argmax(y_true, axis=-1)
        y_pred_class = acc.argmax(y_pred, axis=-1)
 
        ignore_mask = acc.cast(acc.not_equal(y_pred_class, to_ignore), 'int32')
        matches = acc.cast(acc.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = acc.sum(matches) / acc.maximum(acc.sum(ignore_mask), 1)
        return accuracy
    return ignore_accuracy


tagged_sentences = br.tagged_sents(tagset ="universal")
sentences , sent_tags = [],[]
for tag_sent in tagged_sentences:
    sent, tags = zip(*tag_sent)
    sentences.append(np.array(sent))
    sent_tags.append(np.array(tags))

 

X_train,X_test,y_train,y_test = train_test_split(sentences, sent_tags, test_size=0.2)
words, tags = set([]), set([])
 
for sents in X_train:
    for x in sents:
        words.add(x.lower())
        
 
for tsents in y_train:
    for t in tsents:
        tags.add(t)
 
    
def scalarfunc(data):
    sum_data = sum(data)
    count_data = len(data)
    scaled_d =[]
    mean = sum_data/count_data
    std = np.std(data)
    for x in data:
        scaled_d.append((x-mean)/math.sqrt(std))
    return scaled_d


def scalerfunc(data):
    min_data = min(data)
    max_data = max(data)
    col =[]
    for x in data:
        col.append((x-min_data)/(max_data-min_data))
    return col


'''
listing = [i for i in range(0,len(words)+2)]
values = scalerfunc(listing)
windex = {w: values[i + 2] for i, w in enumerate(list(words))}
windex['-PADS-'] = values[0]
windex['-NA-'] = values[1]

listing = [i for i in range(0,len(tags)+2)]
value = scalerfunc(listing)
tindex = {t: value[i + 1] for i, t in enumerate(list(tags))}
tindex['-PADS-'] = value[0]
'''

windex = {w: i + 2 for i, w in enumerate(list(words))}
windex['-PADS-'] = 0
windex['-NA-'] = 1
 
tindex = {t: i + 1 for i, t in enumerate(list(tags))}
tindex['-PADS-'] = 0




train_X, test_X, train_y, test_y = [], [], [], []
 
for sents in X_train:
    sindex = []
    for x in sents:
        try:
            sindex.append(windex[x.lower()])
        except KeyError:
            sindex.append(windex['-NA-'])
 
    train_X.append(sindex)
 
for sents in y_test:
    sindex = []
    for x in sents:
        try:
            sindex.append(windex[x.lower()])
        except KeyError:
            sindex.append(windex['-NA-'])
 
    test_X.append(sindex)
 
for sents in y_train:
    train_y.append([tindex[t] for t in sents])
 
for sents in y_test:
    test_y.append([tindex[t] for t in sents])
MAX_LEN = len(max(train_X,key=len))
    
train_X = pad_sequences(train_X, maxlen=MAX_LEN, padding='post')
test_X = pad_sequences(test_X, maxlen=MAX_LEN, padding='post')
train_y = pad_sequences(train_y, maxlen=MAX_LEN, padding='post')
test_y = pad_sequences(test_y, maxlen=MAX_LEN, padding='post')
  

model = bi_lstm_model()
 
model.summary() 



def hotencoding(t_data, tag_c):
    train_y = []
    for s in t_data:
        tags_v = []
        for item in s:
            tags_v.append(np.zeros(tag_c))
            tags_v[-1][item] = 1.0
        train_y.append(tags_v)
    return np.array(train_y)


model.fit(train_X, hotencoding(train_y, len(tindex)), batch_size=128, epochs=50, validation_split=0.2)
def reverse(pred, index):
    token_seq = []
    for t_seq in pred:
        token = []
        for tag_c in t_seq:
            token.append(index[np.argmax(tag_c)])
        token_seq.append(token)
    return token_seq

predictions = model.predict(test_X)


pred = reverse(predictions, {i: t for t, i in tindex.items()})

classification_report = classification_report(y_test,pred)
accuracy = accuracy_score(y_test,pred)
confusion_matrix = confusion_matrix(y_test,pred)
print(classification_report)
print(accuracy)
print(confusion_matrix)

 



    
    


    
    
