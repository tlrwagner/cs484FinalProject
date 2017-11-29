import csv
import requests
# import poetrytools
from gensim.models import Word2Vec
# import word2vec
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, BayesianRidge, Lasso

stop_words = stopwords.words('english')
lemmtizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer('[A-Za-z]\w+')
num_features = 500

class Artist:
    def __init__(self, name):
        self.name = name
        self.songs = []
        return

    def add_song(self, song):
        self.songs.append(song)

    def __str__(self):
        return "%s\n%r" % (self.name, self.songs)


class Song:
    def __init__(self, lyrics, title, artist):
        self.lyrics = lyrics
        self.title = title
        return

    def __str__(self):
        return self.title

def get_artists_songs():
    with open('Lyrics2.csv', 'rb') as in_file:
        i = 0
        artists = []
        result_artists = []
        songs = []
        reader = csv.reader(in_file)
        for line in reader:
            artists.append(line[0])
            songs.append(Song(line[1], line[2], line[0]))
            i += 1
        print(i)

        for artist in list(set(artists)):
            result_artists.append(Artist(artist))

    # for artist in result_artists:
    #     print artist
    # for song in songs:
    #     # print song
    #     pass

def get_spooky():
    with open('spooky_train.csv', 'rb') as infile:
        reader = csv.reader(infile)
        i = 0
        for line in reader:
            if i == 1:
                print (line[1])
                print (line[2])
            # print line[1]
                pass
            i += 1
        print (i)

# with open('all.csv', 'rb') as infile:
#     reader = csv.reader(infile)
#     i = 0
#     for line in reader:
#         if i == 1:
#             pass
#             # print line[1]
#             # print line[2]
#             # print line[1]
#         i += 1
#     print i

def get_poemdb():
    r = requests.get('http://poetrydb.org/author/%20/author')
    num_authors = {}
    for item in r.json():
        author = item['author']
        if item['author'] not in num_authors.keys():
            num_authors[item['author']] = 1
        else:
            num_authors[item['author']] += 1
        # print item['author']
    print (num_authors)

# get_poemdb()

sentences = ["this is a sentence", "another sentence", "also one more"]

# word2vec.doc2vec('spooky_train.csv', 'testfile.bin')
# model = word2vec.load('testfile.bin')
# print model.vocab

def get_stemmed_tokens_from_csv(infile):
    train = pd.read_csv(infile)

    total_word_count = 0
    nonstop_count = 0
    data = []
    for sentence in train.text.values:
        # print(sentence)
        sentence_tokens = []
        for word in tokenizer.tokenize(sentence):
            total_word_count += 1
            if word.lower() not in stop_words:
                nonstop_count += 1
                sentence_tokens.append(word.lower())
        data.append(sentence_tokens)
    # print(total_word_count)
    # print(nonstop_count)
    # print(data)
    return data

def get_feature_vector(tokens, num_features, w2v_model):
    feature_vector = np.zeros(shape=(1, num_features), dtype='float32')
    missed = 0
    for word in tokens:
        try:
            featureVector = np.add(featureVector, w2v_model[word])
        except:
            missed += 1
            pass
    if len(tokens) - missed == 0:
        return np.zeros(shape=(num_features), dtype='float32')
    return np.divide(featureVector, len(tokens) - missed).squeeze()

def get_vectors(in_data, in_model):
    result_vectors = []
    for token_sentence in in_data:
        ftr_vector = get_feature_vector(token_sentence, num_features, in_model)
        result_vectors.append(ftr_vector)
    return result_vectors


def spooky_get_predictions_from_csv():
    in_file = 'test.csv'
    data = get_stemmed_tokens_from_csv(in_file)
    model = Word2Vec(data, size=num_features)
    vectors = get_vectors(data, model)
    print('finished getting vectors for test data')
    predict_estimator = spooky_get_estimator()
    print('finished getting estimator')
    # probabilities = predict_estimator.predict_proba(vectors)
    # print(vectors)
    # with open('sampleout.txt','w') as outfile:
    #     for item in vectors:
    #         outfile.write(str(item.tostring()))
    probabilities = predict_estimator.predict(vectors)
    return probabilities


def spooky_get_estimator():
    train = pd.read_csv('spooky_train.csv')
    data = get_stemmed_tokens_from_csv('spooky_train.csv')
    model = Word2Vec(data, size=num_features)
    vectors = get_vectors(data, model)

    train['author'] = train['author'].map({'EAP': 0, 'HPL': 1, 'MWS': 2})

    # estimator = LogisticRegression(C=1)
    # estimator = LinearRegression()
    estimator = Lasso()
    estimator.fit(np.array(vectors), train.author.values)
    return estimator

probs = spooky_get_predictions_from_csv()

# print(probs)
# author = pd.DataFrame(probs)
#
# test = pd.read_csv('test.csv')
#
# final = pd.DataFrame()
# final['id'] = test.id
# final['EAP'] = author[0]
# final['HPL'] = author[1]
# final['MWS'] = author[2]
# final.to_csv('submission.csv', sep=',',index=False)


# print(model.most_similar('great'))