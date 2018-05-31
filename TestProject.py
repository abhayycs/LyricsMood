# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# import numpy as np
# from sklearn.pipeline import Pipeline 
# from nltk import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer
# from scipy.sparse import csr_matrix
# from sklearn.tree import DecisionTreeClassifier

import os
import re
import operator
import pandas as pd 
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn import preprocessing

# set display terminal width
pd.set_option('display.width', 192, 'display.max_rows', None)


# constants
SENTIMENT = 'sentiment'
TEXT = 'text'
POSITIVE = 'positive'
NEGATIVE = 'negative'
NEUTRAL = 'neutral'
SPLIT_LINE = '\n'
SPLIT_WORD = ' '
KEYWORD = 'hypernym'
FREQUENCY  = 'freq'
FEATURE_KEY = [KEYWORD, FREQUENCY]

stop = list(set(stopwords.words('english'))) # stopwords

class SongSentiment(object):
    '''
    Generic Twitter Class for sentiment analysis.
    '''
    def __init__(self, train, test):
        '''
        Class constructor or initialization method.
        '''
        self.MOOD = 'mood'
        self.FILENAME = 'filename'
        self.LYRICS = 'lyrics'
        self.HYPERNYMS = 'hypernyms'
        self.ENCODING = 'ISO-8859-1'
        self.FREQ = 3

        self.train = train
        self.test = test

    def preprocessing_df(self, moods):
        print('preprocessing_df')
        self.df = pd.DataFrame(columns=[self.MOOD, self.FILENAME])

        for mood in moods:
            for song_type, dirs, filenames in self.find_songs(mood):
                for filename in filenames:
                    self.df = self.df.append(pd.Series([song_type, filename], index=self.df.columns), ignore_index=True)

        self.df[self.LYRICS] = self.df[[self.MOOD, self.FILENAME]].apply(lambda x: self.read_content(os.path.join(*x)), axis=1)
        # self.df[HYPERNYMS] = self.df[[LYRICS]].apply(lambda x: self.lyrics_analyzer(*x), axis=1)
        self.df[self.HYPERNYMS] = self.df[[self.LYRICS]].apply(lambda x: self.reduce_noise(self.lyrics_analyzer(*x), 1, 20, 200), axis=1)
        # self.df[['positiveness','negativeness']] = self.df[['lyrics']].apply(lambda x: self.overall_lyrics_mood(*x), axis=1)
        
    def train_model_df(self):
        print('train_model_df')
        # create train model dataframe
        for i in self.df.index:
            self.train_df = self.df[self.HYPERNYMS].apply(pd.Series)
        # fill all NaN values to '0'
        self.train_df.fillna(0, inplace=True)
        for column in self.train_df.columns:
            if self.train_df[column].sum() <= self.FREQ:
                self.train_df = self.train_df.drop(column, axis = 1)
        self.features = list(self.train_df.columns)

    def test_model_df(self):
        print('test_model_df')
        # create test model dataframe
        for i in self.df.index:
            self.test_df = self.df[self.HYPERNYMS].apply(pd.Series)
        new_columns = set(self.features) - set(self.test_df.columns)
        for column in new_columns:
            self.test_df[column] = 0

        # fill all NaN values to '0'
        self.test_df.fillna(0, inplace=True)

    def predict(self):
        '''
        '''
        print('predict')
        self.result = self.clf.predict(self.test_df[self.features])

    # recursively list all song files
    def find_songs(self, path):
        '''
        '''
        for root, dirs, files in os.walk(path): 
            yield(root, dirs, files)

    def read_content(self, file_path):
        '''
        '''
        return open(file=file_path, mode='r', encoding=self.ENCODING).read()

    def clean_stanza(self, stanza):
        '''
        Utility function to clean stanza text by removing links, special characters
        using simple regex statements.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", stanza).split())

    def lyrics_analyzer(self, fetched_lyrics, flag = 0, pp = 0):
        '''
        '''
        # empty dictionary to store required params of a lyrics
        # expanding vocabulary: Hypernym approach.
        lyrics_hypernyms = {}

        # iterate over stanza one by one
        for stanza in fetched_lyrics.split('\n\n'):
            if stanza != '':
                # iterate over line by line
                for line in self.clean_stanza(stanza).split(SPLIT_LINE):
                    # iterate over word by word in a line
                    for word in line.split(SPLIT_WORD):
                        word = word.lower()
                        if word not in stop:
                            if pp: print('+', word)        # word is being used
                            # find synonyms sets
                            for synset in wordnet.synsets(word):
                                # iterate over synset hypernyms
                                for hypernym in synset.hypernyms():
                                    # extract hypernym name 
                                    hypernym_0 = hypernym.name().split('.')[0]
                                    # if hypernym name already exists then increase the count value
                                    if hypernym_0 in lyrics_hypernyms.keys():
                                        lyrics_hypernyms[hypernym_0] += 1
                                    else:
                                        lyrics_hypernyms[hypernym_0] = 1
                        else:
                            if pp: print('-', word)        # word is not being used
        # returning 
        if not flag:
            return lyrics_hypernyms
        else:
            return [[word[0],word[1]] for word in sorted(lyrics_hypernyms.items(), key=operator.itemgetter(1), reverse=True)]
    
    def reduce_noise(self, lyrics_hypernyms, min_count, max_count, length, flag = 0):
        '''
        '''
        frequent_hypernyms = {}
        for key in lyrics_hypernyms.keys():
            if (lyrics_hypernyms[key] >= min_count) & (lyrics_hypernyms[key] <= max_count):
                frequent_hypernyms[key]  = lyrics_hypernyms[key]
            # returning 
        if not flag:
            return frequent_hypernyms
        else:
            return [[word[0],word[1]] for word in sorted(frequent_hypernyms.items(), key=operator.itemgetter(1), reverse=True)][:length]

    def overall_lyrics_mood(self, fetched_lyrics):
        '''
        Main function to fetch lyrics and parse them.
        '''
        # empty list to store parsed lyrics
        lyrics = []
        
        # parsing lyrics one by one
        for stanza in fetched_lyrics.split('\n\n'):
            # create TextBlob object of passed stanza text
            analysis = TextBlob(self.clean_stanza(stanza), analyzer=NaiveBayesAnalyzer())
            print(analysis.sentiment)

        # return parsed lyrics
        return 

    def run_ml_classifier(self):
        self.preprocessing_df(self.train)
        # self.features_selection()
        self.train_model_df()
        # train model
        self.clf = MultinomialNB()
        self.clf.fit(self.train_df, self.df['mood'])
        self.preprocessing_df(self.test)
        self.test_model_df()
        self.predict()
        
        # additional task to print the dataframe
        self.df = self.df.drop([self.LYRICS, self.HYPERNYMS], axis=1)
        self.df[self.MOOD] = self.result
        print(self.df)

        return pd.Series(self.result)

def main():
    # creating object of SongSentiment Class
    song = SongSentiment(['Sadness', 'Anger'], ['MoodPredict'])
    song.run_ml_classifier()

if __name__ == "__main__":
    # calling main function
    main()
