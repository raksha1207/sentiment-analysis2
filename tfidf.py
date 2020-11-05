#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 13:02:12 2020

@author: devikrishnan
"""
#import necessary libraries/models
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier


#vectorize text using TF-IDF
def tfidf_vectorize(tokenized_sent):
    #dummy tokenizer
    def tokenizer(text):
        return text
    
    vectorizer = TfidfVectorizer(tokenizer=tokenizer, lowercase=False, norm=False, smooth_idf=False)
    vectors_list = vectorizer.fit_transform(tokenized_sent)
    vectors = list(vectors_list.toarray())
    return vectors


#Random forest classifier
def randomforest(X_train, y_train, X_test, y_test):
    model1 = RandomForestClassifier(n_estimators=200, random_state=0)
    model1.fit(X_train, y_train)
    y_pred = model1.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred, average='micro')
    print('Accuracy: '+str(round(accuracy,3)))
    print('F1-score: '+ str(round(f1score, 3)))
    print(classification_report(y_test, y_pred))
    return accuracy, f1score, y_pred


if __name__ == '__main__':
    #read datasets
    traindata = pd.read_excel('P1_training.xlsx')
    testdata = pd.read_excel('P1_testing.xlsx')
    data = pd.concat([traindata, testdata])
    
    #tokenize text
    sentences = data['sentence'].tolist()
    tokenized_sent = [word_tokenize(sentence) for sentence in sentences]
       
    vectors = tfidf_vectorize(tokenized_sent)
       
    #store vectors in dataset
    data['sentence_vectors'] = vectors
    
    #Split dataset into training/testing sets
    traindata = data[:1660]
    testdata = data[1660:]
    X_train = traindata['sentence_vectors'].tolist()
    y_train = traindata['label'].tolist()
    X_test = testdata['sentence_vectors'].tolist()
    y_test = testdata['label'].tolist()
    #uncomment below line if you want to use only one dataframe without splitting manually. testsize = 0.33 for example.
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    #classify sentences
    accuracy, f1, predictions = randomforest(X_train, y_train, X_test, y_test)
    testdata['predicted_label'] = predictions
    testdata.rename(columns={'label': 'gold_label'}, inplace=True)
    testdata.to_csv('testing_output_tfidf.csv', columns=(['sentence','gold_label','predicted_label']))
    




















