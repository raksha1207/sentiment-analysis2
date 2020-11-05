#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 14:18:02 2020

@author: devikrishnan
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sentence_transformers import SentenceTransformer


#split dataset into train/test sets
def splitdata(data, ts):
    X = data['sentence']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)
    return X_train, X_test, y_train, y_test


#vectorizer
def bert_vectorize(sentences):
    #load pretrained BERT model
    model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    #encode sentences
    vectors = model.encode(sentences)
    return list(vectors)


#Random forest classifier
def randomforest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred, average='micro')
    print('Accuracy: '+str(round(accuracy,3)))
    print('F1-score: '+ str(round(f1score, 3)))
    print(classification_report(y_test, y_pred))
    return accuracy, y_pred


#perform sentiment classification of sentences
def sentimentAnalysis(data, data_len):
    #vectorize sentences
    sentences = data['sentence'].tolist()
    print("Running bert_vectorize...")
    sentence_vectors = bert_vectorize(sentences)
    #print(len(sentence_vectors[0]))
    
    #store vectors in dataset
    data['sentence_vectors'] = sentence_vectors
    
    #split data into training and testing sets
    train = data[:data_len]
    test = data[data_len:] 
    X_train = train['sentence_vectors'].tolist()
    y_train = train['label'].tolist()
    X_test = test['sentence_vectors'].tolist()
    y_test = test['label'].tolist()
    #uncomment below line if you want to use only one dataframe without splitting manually. testsize = 0.33 for example.
    #X_train, X_test, y_train, y_test = splitdata(data,testsize)
    
    #classify data
    accuracy, predictions = randomforest(X_train, y_train, X_test, y_test)
    return accuracy, predictions
    


if __name__ == '__main__':
    #read datasets
    traindata = pd.read_excel('P1_training.xlsx')
    testdata = pd.read_excel('P1_testing.xlsx')
    data = pd.concat([traindata, testdata])
    
    #classify sentences
    accuracy, predictions = sentimentAnalysis(data, len(traindata))
    testdata['predicted_label'] = predictions
    testdata.rename(columns={'label': 'gold_label'}, inplace=True)
    testdata.to_csv('testing_output_bert.csv', columns=(['sentence','gold_label','predicted_label']))
