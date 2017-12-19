import csv
import requests
# import poetrytools
from gensim.models import Word2Vec
# import word2vec
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.linear_model import LogisticRegression, LinearRegression, BayesianRidge, Lasso
from sklearn.tree import DecisionTreeClassifier
import string

import seaborn as sns
import matplotlib.pyplot as plt

stop_words = stopwords.words('english')
lemmtizer = WordNetLemmatizer()
port_stemmer = PorterStemmer()
tokenizer = RegexpTokenizer('[A-Za-z]\w+')
num_features = 150

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
            word = port_stemmer.stem(word)
            if word.lower() not in stop_words:
                nonstop_count += 1
                sentence_tokens.append(word.lower())
        data.append(sentence_tokens)
    # print(total_word_count)
    # print(nonstop_count)
    # print(data)
    return data

def get_feature_vector(tokens, num_features, w2v_model):
    featureVec = np.zeros(shape=(1, num_features), dtype='float32')
    missed = 0
    success = 0
    for word in tokens:
        # print(word)
        # featureVec = np.add(featureVec, w2v_model.wv[word])


        try:
            # print(w2v_model[word])
            # print("word: " + word)term
            featureVec = featureVec + w2v_model[word]
            print("word: " + word)
            # featureVector = np.add(featureVector, w2v_model[word])
            # print(featureVec)
            success += 1
        except:
            missed += 1
            pass
    print('success: %d missed: %d' % (success, missed))
    if len(tokens) - missed == 0:
        return np.zeros(shape=(num_features), dtype='float32')
    return (featureVec / len(tokens) - missed).squeeze()

def get_vectors(in_data, in_model):
    result_vectors = []
    for token_sentence in in_data:
        ftr_vector = get_feature_vector(token_sentence, num_features, in_model)
        result_vectors.append(ftr_vector)
    return result_vectors


def spooky_get_predictions_from_csv_w2v():
    in_file = 'test.csv'
    data = get_stemmed_tokens_from_csv(in_file)
    model = Word2Vec(data, size=num_features)
    vectors = get_vectors(data, model)
    print('finished getting vectors for test data')
    predict_estimator = spooky_get_estimator_w2v()
    print('finished getting estimator')
    probabilities = predict_estimator.predict_proba(vectors)
    # probabilities = predict_estimator.predict(vectors)
    return probabilities



def spooky_get_estimator_w2v():
    train = pd.read_csv('spooky_train.csv')
    data = get_stemmed_tokens_from_csv('spooky_train.csv')
    model = Word2Vec(data, size=num_features, min_count=3, window=5, sg=1, alpha=1e-4, workers=4)
    vectors = get_vectors(data, model)

    train['author'] = train['author'].map({'EAP': 0, 'HPL': 1, 'MWS': 2})

    estimator = LogisticRegression(C=1)
    # estimator = LinearRegression()
    # estimator = Lasso()
    estimator.fit(np.array(vectors), train.author.values)
    return estimator


def spooky_output_word2vec():
    probs = spooky_get_predictions_from_csv_w2v()

    author = pd.DataFrame(probs)
    #
    test = pd.read_csv('test.csv')
    #
    final = pd.DataFrame()
    final['id'] = test.id
    final['EAP'] = author[0]
    final['HPL'] = author[1]
    final['MWS'] = author[2]
    final.to_csv('submission.csv', sep=',', index=False)


def spooky_get_estimator_tfidf():
    train = pd.read_csv('spooky_train.csv')
    data = get_stemmed_tokens_from_csv('spooky_train.csv')
    data2 = []
    for i in range(len(data)):
        data2.append(" ".join(data[i]))
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    vector_train = vectorizer.fit_transform(data2)
    # vectors = get_vectors(data, model)
    nb_model = MultinomialNB()
    svc_model = LinearSVC()

    nb_model.fit(vector_train, train['author'])


    # print(train['author'])

    # train['author'] = train['author'].map({'EAP': 0, 'HPL': 1, 'MWS': 2})


    # estimator = LogisticRegression(C=1)
    # estimator = LinearRegression()
    # estimator = Lasso()
    # estimator.fit(np.array(vectors), train.author.values)
    # return estimator
    return

def get_stats():
    train = pd.read_csv('spooky_train.csv')
    train['num_words'] = train['text'].apply(lambda x: len(str(x).split()))
    train['num_char'] = train['text'].apply(lambda x: len(x))
    train['punctuation'] = train['text'].apply(lambda x: len([char for char in str(x) if char in string.punctuation]))
    # author_num_words = {'EAP':[], 'MWS':[], 'HPL':[]}
    # for i in range(len(train['text'])):
    #     author_num_words[train['author'][i]].append(len(train['text'][i]))
    # print(author_num_words)
    # data = seaborn.load_dataset(train)
    # print(train.head())
    sns.violinplot(x='author', y='punctuation', data=train)
    plt.ylim(0,20)
    plt.show()

def spooky_get_estimator_tfidf_test():
    train = pd.read_csv('spooky_train.csv')
    data = get_stemmed_tokens_from_csv('spooky_train.csv')
    data2 = []
    for i in range(len(data)):
        data2.append(" ".join(data[i]))
    k = 5
    kf = KFold(n_splits=k)
    nb_accuracy_sum = 0
    svc_accuracy_sum = 0
    nb_f1_sum = 0
    svc_f1_sum = 0
    # nb_logloss_sum = 0
    # svc_logloss_sum = 0
    for testing, training in kf.split(data2):
        train_data = np.array(data2)[training]
        test_data = np.array(data2)[testing]
        train_data = train_data.tolist()
        test_data = test_data.tolist()
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        # print(train_data)
        vector_train = vectorizer.fit_transform(train_data)
        vector_test = vectorizer.transform(test_data)
        ytrain, ytest = train['author'][training], train['author'][testing]
        # vectors = get_vectors(data, model)
        nb_model = MultinomialNB()
        svc_model = LinearSVC()
        # nb_model = BernoulliNB()
        # svc_model = NuSVC()
        # nb_model = GaussianNB()
        # svc_model = SVC()


        # print(len(ytrain))
        # print(len(ytest))
        nb_model.fit(vector_train.toarray(), ytrain)
        svc_model.fit(vector_train, ytrain)
        # nb_result = nb_model.predict(vector_test)
        # svc_result = svc_model.predict(vector_test)
        nb_result = nb_model.predict(vector_test.toarray())
        svc_result = svc_model.predict(vector_test)

        nb_score = metrics.precision_score(ytest, nb_result, average='weighted')
        svc_score = metrics.precision_score(ytest, svc_result, average='weighted')
        nb_f1 = metrics.recall_score(ytest, nb_result, average='weighted')
        svc_f1 = metrics.recall_score(ytest, svc_result, average='weighted')

        # nb_score = metrics.accuracy_score(ytest, nb_result)
        # svc_score = metrics.accuracy_score(ytest, svc_result)
        # nb_f1 = metrics.f1_score(ytest, nb_result, average='weighted')
        # svc_f1 = metrics.f1_score(ytest, svc_result, average='weighted')

        # nb_logloss = metrics.log_loss(ytest, nb_result, normalize=True)
        # svc_logloss = metrics.log_loss(ytest, svc_result)
        nb_f1_sum += nb_f1
        svc_f1_sum += svc_f1
        nb_accuracy_sum += nb_score
        svc_accuracy_sum += svc_score
        # nb_logloss_sum += nb_logloss
        # svc_logloss_sum += svc_logloss

        # print("Naive Bayes:     " + str(nb_score))
        # print("SVC:             " + str(svc_score))
    nb_accuracy_sum /= k
    svc_accuracy_sum /= k
    nb_f1_sum /= k
    svc_f1_sum/= k
    # nb_logloss_sum /= k
    # svc_logloss_sum /= k
    print("NB avg accuracy:         " + str(nb_accuracy_sum))
    print("SVC avg accuracy:        " + str(svc_accuracy_sum))
    print("NB avg f1:               " + str(nb_f1_sum))
    print("SVC avg f1:              " + str(svc_f1_sum))
    # print("NB avg log-loss:         " + str(nb_logloss_sum))
    # print("SVC avg log-loss:        " + str(svc_logloss_sum))




    # nb_model.fit(vector_train, train['author'])

def spooky_get_predictions_from_csv_tfidf():
    in_file = 'test.csv'
    data = get_stemmed_tokens_from_csv(in_file)
    model = Word2Vec(data, size=num_features)
    vectors = get_vectors(data, model)
    print('finished getting vectors for test data')
    predict_estimator = spooky_get_estimator_w2v()
    print('finished getting estimator')
    probabilities = predict_estimator.predict_proba(vectors)
    # probabilities = predict_estimator.predict(vectors)
    return probabilities

# train = pd.read_csv('spooky_train.csv')
# data = get_stemmed_tokens_from_csv('spooky_train.csv')
# model = Word2Vec(data, size=num_features)
# print(model.most_similar('raven'))
# print(len(model.wv.vocab))


# spooky_output_word2vec()

spooky_get_estimator_tfidf_test()
# get_stats()

# print(model.most_similar('great'))