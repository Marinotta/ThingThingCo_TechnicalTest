# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 13:47:36 2022

@author: Marina Lacambra
"""

import preprocessor as p
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score


def tweet_cleaner(tweet):
    #Removes hashtags
    clean_tweet = p.clean(tweet)
    return clean_tweet

def tweet_tokenizer(clean_tweet):
    #Tokenizes the tweet
    tknzd_clean_tweet = nltk.word_tokenize(clean_tweet)
    return tknzd_clean_tweet
    
def tweet_stopword_remover(tknzd_clean_tweet):
    #R
    tknzd_words_tweet = [word for word in tknzd_clean_tweet if word.lower() not in stopwords.words('english')]
    return tknzd_words_tweet

def tweet_punct_remover(tknzd_words_tweet):
    clean_tokens = [t for t in tknzd_words_tweet if re.match(r'[^\W\d]*$', t)]
    return (clean_tokens)

def lemmatizer(tknzd_word_tweet):
    lem = WordNetLemmatizer()
    normalized_tweet = []
    for word in tknzd_word_tweet:
        normalized_text = lem.lemmatize(word, 'v')
        normalized_tweet.append(normalized_text)
    return normalized_tweet

def joiner (lemmatized_tweet):
    clean_tweets = [' '.join(lemmatized_tweet)]
    return clean_tweets

def tweet_dataset_cleaner_tocsv(input_csvfile, output_csvfile):
    "the column containing the tweets must be named 'tweet'"
    tweet_dataset = pd.read_csv(input_csvfile)
    clean_tweet = [tweet_cleaner(tweet) for tweet in tweet_dataset['tweet']]
    tknzd_clean_tweet = [tweet_tokenizer(tweet) for tweet in clean_tweet]
    tknzd_words_tweet = [tweet_stopword_remover(tweet) for tweet in tknzd_clean_tweet]
    tknzd_nopunct_tweet = [tweet_punct_remover(tweet) for tweet in tknzd_words_tweet]
    tknzd_lemmas = [lemmatizer(tweet) for tweet in tknzd_nopunct_tweet]
    clean_tweets = [joiner(tweet) for tweet in tknzd_lemmas]
    tweet_dataset['tweet'] = clean_tweets
    tweet_dataset.to_csv(output_csvfile, index=False, header=True)    
    print ('Clean dataset available in' , output_csvfile)
    
def tweet_dataset_cleaner(input_csvfile):
    "the column containing the tweets must be named 'tweet'"
    tweet_dataset = pd.read_csv(input_csvfile)
    clean_tweet = [tweet_cleaner(tweet) for tweet in tweet_dataset['tweet']]
    tknzd_clean_tweet = [tweet_tokenizer(tweet) for tweet in clean_tweet]
    tknzd_words_tweet = [tweet_stopword_remover(tweet) for tweet in tknzd_clean_tweet]
    tknzd_nopunct_tweet = [tweet_punct_remover(tweet) for tweet in tknzd_words_tweet]
    tknzd_lemmas = [lemmatizer(tweet) for tweet in tknzd_nopunct_tweet]
    clean_tweets = [joiner(tweet) for tweet in tknzd_lemmas]
    tweet_dataset['tweet'] = clean_tweets
    return (tweet_dataset)

def train_classifier(input_csvfile):
    "the column containing the tweets must be named 'tweet', the column containing the labels must be named 'label'"
    tweet_dataset = pd.read_csv(input_csvfile)
    msg_train, msg_test, label_train, label_test = train_test_split(tweet_dataset['tweet'], tweet_dataset['label'], test_size=0.2)
    pipeline = Pipeline([
        ('bow',CountVectorizer()),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])
    pipeline.fit(msg_train,label_train)
    predictions = pipeline.predict(msg_test)
    print(classification_report(predictions,label_test))
    print ('\n')
    print(confusion_matrix(predictions,label_test))
    print(accuracy_score(predictions,label_test))
    

def bonus():
    input_csvfile = input("Enter path of your csv testing sample:")
    test = pd.read_csv(input_csvfile)
    clean_test = tweet_dataset_cleaner(test)
    tweet_train_dataset = pd.read_csv("https://raw.githubusercontent.com/MohamedAfham/Twitter-Sentiment-Analysis-Supervised-Learning/master/Data/train_tweets.csv")
    clean_train = tweet_dataset_cleaner(tweet_train_dataset)
    msg_train, msg_test, label_train, label_test = train_test_split(clean_train['tweet'], clean_train['label'], test_size=0.2)
    pipeline = Pipeline([
    ('bow',CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
    pipeline.fit(msg_train,label_train)
    predictions = pipeline.predict(clean_test["tweet"])
    tweet_list = [tweet for tweet in clean_test["tweet"]]
    for i in range(len(predictions)):
        print ((tweet_list[i],predictions[i]))