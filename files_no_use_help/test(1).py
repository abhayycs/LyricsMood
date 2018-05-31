import os
import re
import operator
import pandas as pd 
import numpy as np
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline 
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from scipy.sparse import csr_matrix
from sklearn.tree import DecisionTreeClassifier
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
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

class SongSentiment(object):
    '''
    Generic Twitter Class for sentiment analysis.
    '''
    def __init__(self):
        '''
        Class constructor or initialization method.
        '''

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

    def get_stanza_sentiment(self, stanza):
        '''
        Utility function to classify sentiment of passed stanza
        using textblob's sentiment method
        '''
        # create TextBlob object of passed stanza text
        analysis = TextBlob(self.clean_stanza(stanza), analyzer=NaiveBayesAnalyzer())
        print(analysis.sentiment)
        # set sentiment
        if (analysis.sentiment.polarity > 0) & (analysis.sentiment.subjectivity > 0):
            return POSITIVE
        elif (analysis.sentiment.polarity < 0)  & (analysis.sentiment.subjectivity > 0):
            return NEGATIVE
        elif (analysis.sentiment.polarity == 0)  & (analysis.sentiment.subjectivity > 0):
            return NEUTRAL
        else:
            return None
 
    def get_lyrics(self, fetched_lyrics):
        '''
        Main function to fetch lyrics and parse them.
        '''
        # empty list to store parsed lyrics
        lyrics = []
        
        # unique_hypernyms
        unique_hypernyms = set()
        print('The Song lyrics:', type(fetched_lyrics), '\n', fetched_lyrics)

        # parsing lyrics one by one
        for stanza in fetched_lyrics.split('\n\n'):
            # empty dictionary to store required params of a stanza
            parsed_stanza = {}
            print('-------------------------------------------------')

            # saving text of stanza
            parsed_stanza[TEXT] = stanza
            print('saving text of stanza:', parsed_stanza[TEXT])

            # saving sentiment of stanza
            parsed_stanza[SENTIMENT] = self.get_stanza_sentiment(stanza)
            print('saving sentiment of stanza:', parsed_stanza[SENTIMENT]) 

            # if stanza has relyrics, ensure that it is appended only once
            if parsed_stanza not in lyrics:
                lyrics.append(parsed_stanza)
            # else:
            #     lyrics.append(parsed_stanza)
            print('-------------------------------------------------')

        # return parsed lyrics
        return lyrics
    def label_encoder(self, key_list):
        '''
        '''
        self.le = preprocessing.LabelEncoder()
        self.le.fit(key_list)

    def label_transform(self, key_list):
        '''
        '''
        try:
            self.le
        except:
            self.le = preprocessing.LabelEncoder()
            self.le.fit(key_list)
        return self.le.transform(key_list)
    def sparse_matrix(self, freq_hypernyms):
        '''
        '''
        data = freq_hypernyms[0]

        return csr_matrix((data, (row, col))).toarray()

def main():
    # creating object of SongSentiment Class
    song = SongSentiment()
    # path = 'Sadness'
    # filename = 'cheap_thrills_sia.txt'

    # file_path = os.path.join(path, filename)
    # # text lyrics
    # song_lyrics = open(file=file_path, mode='r', encoding='ISO-8859-1').read()
    # print('------------------------------------------------------------------------------')
    # # calling function to hypernyms
    # frequent_hypernyms = song.lyrics_analyzer(song_lyrics)
    
    # print('------------------------------------------------------------------------------')
    # # clean hypernyms data
    # frequent_hypernyms = song.reduce_noise(frequent_hypernyms, 2, 100, 200)
    
    # print('------------------------------------------------------------------------------')
    # text_clf = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB(alpha=0.1))])
    # # text_clf = Pipeline([('vect', TfidfVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB(alpha=0.1))])
 
    SONG_TYPE = ['Sadness', 'Anger']

    # recursively list all song files
    def find_songs(path): 
        for root, dirs, files in os.walk(path): 
            yield(root, dirs, files)

    def read_content(file_path):
        print(file_path)
        return open(file=file_path, mode='r', encoding='ISO-8859-1').read()

    df = pd.DataFrame(columns=['mood', 'filename'])

    for MOOD in SONG_TYPE:
        for song_type, dirs, filenames in find_songs(MOOD):
            for filename in filenames:
                df = df.append(pd.Series([song_type, filename], index=df.columns), ignore_index=True)


    df['lyrics'] = df[['mood','filename']].apply(lambda x: read_content(os.path.join(*x)), axis=1)
    # df['hypernyms'] = df[['lyrics']].apply(lambda x: song.lyrics_analyzer(*x), axis=1)
    df['hypernyms'] = df[['lyrics']].apply(lambda x: song.reduce_noise(song.lyrics_analyzer(*x), 1, 20, 200), axis=1)
    print(df)
    
    hypernyms_dict = {}
    [hypernyms_dict.update(hypernyms) for hypernyms in df['hypernyms']]
    features = list(hypernyms_dict.keys())
    print(len(hypernyms_dict))
    print(type(features))
    print('----------------------------------------')
    process_df = pd.DataFrame(columns = features)
    for i in df.index:
        x = pd.DataFrame(df['hypernyms'][i], index=[i])
        process_df = process_df.append(x)
    # process_df['mood'] = df['mood']
    process_df.fillna(0, inplace=True)
    print(process_df)

    # clf = tree.DecisionTreeClassifier()
    clf = MultinomialNB()
    clf.fit(process_df, df['mood'])

    SONG_TYPE = ['Test']

    df = pd.DataFrame(columns=['mood', 'filename'])

    for MOOD in SONG_TYPE:
        for song_type, dirs, filenames in find_songs(MOOD):
            for filename in filenames:
                df = df.append(pd.Series([song_type, filename], index=df.columns), ignore_index=True)


    df['lyrics'] = df[['mood','filename']].apply(lambda x: read_content(os.path.join(*x)), axis=1)
    df['hypernyms'] = df[['lyrics']].apply(lambda x: song.lyrics_analyzer(*x), axis=1)
    # df['hypernyms'] = df[['lyrics']].apply(lambda x: song.reduce_noise(song.lyrics_analyzer(*x), 2, 40, 200), axis=1)
    print(df)
    
    process_df = pd.DataFrame(columns = features)
    for i in df.index:
        x = pd.DataFrame(df['hypernyms'][i], index=[i])
        process_df = process_df.append(x)
    # process_df['mood'] = df['mood']
    process_df.fillna(0, inplace=True)
    print(process_df)

    ans = clf.predict(process_df[features])
    print(ans)

    # print('------------------------------------------------------------------------------')
    # song_lyrics = open(file=file_path, mode='r', encoding='ISO-8859-1')
    # clf = TfidfVectorizer()
    # # clf = MultinomialNB()
    # # clf = CountVectorizer()
    # res = clf.fit_transform(song_lyrics)
    # print(res)
    # print(res.shape)
    # print(type(song_lyrics))
    # for i in res:
    #     print(i)
    #     print(type(i))
    #     print(i.shape)

    # print('------------------------------------------------------------------------------')
    # res = text_clf.fit_transform(song_lyrics)
    # print(res)
    # print(res.shape)
    # print(type(song_lyrics))
    # for i in res:
    #     print(i)

    # print('------------------------------------------------------------------------------')
    # X = pd.DataFrame([[ i[0] for i in frequent_hypernyms1], [ i[0] for i in frequent_hypernyms2]])
    # print(X)
    # Y = ['cheap','cold']
    # text_clf.fit_transform(X, Y)
    # ans = text_clf.predict(frequent_hypernyms3)
    # print(ans)
    # print('------------------------------------------------------------------------------')


    # lyrics = song.get_lyrics(song_lyrics)
   
    # # picking positive lyrics from lyrics
    # positive_lyrics = [stanza for stanza in lyrics if stanza[SENTIMENT] == POSITIVE]
    # # percentage of positive lyrics
    # print("Positive lyrics percentage: {} %".format(100*len(positive_lyrics)/len(lyrics)))
    # # picking negative lyrics from lyrics
    # negative_lyrics = [stanza for stanza in lyrics if stanza[SENTIMENT] == NEGATIVE]
    # # percentage of negative lyrics
    # print("Negative lyrics percentage: {} %".format(100*len(negative_lyrics)/len(lyrics)))
    # # picking negative lyrics from lyrics
    # neutral_lyrics = [stanza for stanza in lyrics if stanza[SENTIMENT] == NEUTRAL]
    # # percentage of neutral lyrics
    # print("Neutral lyrics percentage: {} % ".format(100*len(neutral_lyrics)/len(lyrics)))
 
    # # printing first 5 positive lyrics
    # print("\n\nPositive lyrics:")
    # for stanza in positive_lyrics[:5]:
    #     print(stanza['text'])
 
    # # printing first 5 negative lyrics
    # print("\n\nNegative lyrics:")
    # for stanza in negative_lyrics[:5]:
    #     print(stanza['text'])
 
if __name__ == "__main__":
    # calling main function
    main()
