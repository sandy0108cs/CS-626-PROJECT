# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 12:29:14 2020

@author: 91799
"""


import nltk
from nltk.corpus import brown as br
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras import callbacks
from keras.layers import Dense, LSTM,Dropout, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences
from keras import backend as acc
import math


def custom_accuracy():
    def masked_accuracy(y_true, y_pred):
        true_id = acc.argmax(y_true, axis=-1)
        pred_id = acc.argmax(y_pred, axis=-1)
        ignore_mask = acc.cast(acc.not_equal(pred_id,0), 'int32')
        tensor = acc.cast(acc.equal(true_id, pred_id), 'int32') * ignore_mask
        accuracy = acc.sum(tensor) / acc.maximum(acc.sum(ignore_mask), 1)
        return accuracy
    return masked_accuracy



def onehotencoding(tag_data, length):
    train_y = []
    for x in tag_data:
        tag = []
        for i in x:
            zeroes = np.zeros(length)
            zeroes[i] = 1.0
            tag.append(zeroes)
        train_y.append(tag)
    return np.array(train_y)


def bi_lstm_model():    
    model = Sequential()
    model.add(InputLayer(input_shape=(MAX_LEN, )))
    model.add(Embedding(len(windex), 128))
    model.add(Bidirectional(LSTM(128, return_sequences=True,recurrent_dropout=0.2)))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(len(tindex),activation='softmax')))
#    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy', custom_accuracy()])
    return model



def serialize(xyz,pred):
    xyz = xyz.reshape(-1,1).flatten()
    x = pd.Series(np.array(pred).reshape(-1,1).flatten(),name = "Predicted")
    index = {i: t for t, i in tindex.items()}
    s = []
    for i in xyz:
        s.append(index[i])
    s= pd.Series(s,name = "Actual")
    return s,x
def custom_matrix(xyz,pred):
    s,x = serialize(xyz,pred)
    print(pd.crosstab(s,x))


def custom_accuracy_score(xyz,pred):
    s,x = serialize(xyz,pred)
    print(accuracy_score(s,x))
    
    
data_sent = br.tagged_sents(tagset ="universal")
sentences , sent_tags = [],[]
for sent in data_sent:
    sentence, tags = zip(*sent)
    sentences.append(np.array(sentence))
    sent_tags.append(np.array(tags))


X_train,X_test,y_train,y_test = train_test_split(sentences, sent_tags, test_size=0.2)
words, tags = set([]), set([])
 


for sents in X_train:
    for x in sents:      
        words.add(x.lower())

for tsents in y_train:
    for t in tsents:
        tags.add(t)
 
    

windex = {w: i + 2 for i, w in enumerate(list(words))}
windex['PADDING'] = 0
windex['UNKNOWN'] = 1
 
tindex = {t: i + 1 for i, t in enumerate(list(tags))}
tindex['PADDING'] = 0




train_X, test_X, train_y, test_y = [], [], [], []
 
for sents in X_train:
    sindex = []
    for x in sents:
        try:
            sindex.append(windex[x.lower()])
        except KeyError:
            sindex.append(windex['UNKNOWN'])
    train_X.append(sindex)

 
for sents in X_test:
    sindex = []
    for x in sents:
        try:
            sindex.append(windex[x.lower()])
        except KeyError:
            sindex.append(windex['UNKNOWN'])
    test_X.append(sindex)
 
for sents in y_train:
    train_y.append([tindex[t] for t in sents])
 
for sents in y_test:
    test_y.append([tindex[t] for t in sents])
    
MAX_LEN = len(max(train_X,key=len))
Ptrain_sent = pad_sequences(train_X, maxlen=MAX_LEN, padding='post')
Ptest_sent = pad_sequences(test_X, maxlen=MAX_LEN, padding='post')
Ptrain_tag = pad_sequences(train_y, maxlen=MAX_LEN, padding='post')
Ptest_tag = pad_sequences(test_y, maxlen=MAX_LEN, padding='post')
enc_train_tag = onehotencoding(Ptrain_tag, len(tindex))
model = bi_lstm_model()

model.summary()
callback = callbacks.EarlyStopping(monitor='loss')
history = model.fit(Ptrain_sent, enc_train_tag, callbacks=[callback], batch_size=128, epochs=20, validation_split=0.2)

def reverse(pred, index):
    token_seq = []
    for t_seq in pred:
        token = []
        for tag_c in t_seq:
            token.append(index[np.argmax(tag_c)])
        token_seq.append(token)
    return token_seq



predictions = model.predict(Ptest_sent)
pred = reverse(predictions, {i: t for t, i in tindex.items()})

custom_matrix(Ptest_tag,pred)


custom_accuracy_score(Ptest_tag,pred)

print(history.history['accuracy'])

