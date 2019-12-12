import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import bcolz
import pickle
import nltk
from nltk import tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


def parse_n_store_glove(file_path='./data'):
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=f'{file_path}/840B.300d.dat', mode='w')

    with open(f'{file_path}/glove.840B.300d.txt', 'rb') as f:
        # total 2196017 lines
        for l in f:
            line = l.decode().split()
            word = ''.join(line[:-300])
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[-300:]).astype(np.float)
            vectors.append(vect)
            print(idx)
    print("total words: ", len(words))

    vectors = bcolz.carray(vectors[1:].reshape((2196017, 300)), rootdir=f'{file_path}/840B.300d.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(f'{file_path}/840B.300d_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'{file_path}/840B.300d_idx.pkl', 'wb'))


def create_word2GloVe_dict(file_path='./data'):
    vectors = bcolz.open(f'{file_path}/840B.300d.dat')[:]
    words = pickle.load(open(f'{file_path}/840B.300d_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{file_path}/840B.300d_idx.pkl', 'rb'))

    word2GloVe = {w: vectors[word2idx[w]] for w in words}
    pickle.dump(word2GloVe, open(f'{file_path}/word2GloVe_dict.pkl', 'wb'))
    return word2GloVe


def get_glove_embeddings(vocab, word2GloVe):
    n_words = len(vocab)
    emb_dim = 300
    embeddings = np.zeros((n_words, emb_dim))
    words_found = 0
    words_not_found = []

    for i, word in enumerate(vocab):
        try:
            embeddings[i] = word2GloVe[word]
            words_found += 1
        except KeyError:
            words_not_found.append(word)
            embeddings[i] = np.random.normal(scale=0.6, size=(emb_dim,))
    print("words_found:", words_found)

    # print("words not found: ", words_not_found)

    return embeddings


def create_vocab(q1s, q2s, k=None):
    """
        Args: k: k most frequent words
    """
    counter = Counter()
    sentences = q1s + q2s
    for sent in sentences:
        counter.update(sent)
    if k is not None:
        counter = dict(counter.most_common(k))

    # word2idx is dict of (word , word_id)
    word2idx = dict([(word, i + 2) for i, word in enumerate(counter.keys())])
    word2idx['PAD'] = 0
    word2idx['UNK'] = 1
    vocab = ['PAD', 'UNK']
    vocab = vocab + [word for word in counter.keys()]

    with open("vocab.txt", "w") as f:
        f.write(str(vocab))

    return vocab, word2idx


def handle_oov(sents, vocab):
    for sent in sents:
        for i, w in enumerate(sent):
            if w not in vocab:
                sent[i] = 'UNK'

    return sents


def max_sent_len(sents):
    max_len = 0
    lengths = []
    for sent in sents:
        l = len(sent)
        lengths.append(l)
        if l > max_len:
            max_len = l
    return max_len, np.array(lengths)


def sent_word2word_idx(tokenized_sents, word2idx):
    sent_vectors = []
    for sent in tokenized_sents:
        new_sent = [word2idx[w] for w in sent]
        sent_vectors.append(new_sent)
    return sent_vectors


def tokenize_data(sentences):
    # tokenize the dataset
    fdist = FreqDist()
    tokenized_sents = []
    for sentence in sentences:
        tokenized_sent = [w.lower() for w in word_tokenize(sentence)]
        tokenized_sents.append(tokenized_sent)
        fdist.update(tokenized_sent)

    # print("Number of word types in the tokenized data: ", len(fdist))
    return tokenized_sents


def remove_stopwords(tokenized_sents):
    # Removing stop_words using nltk
    stop_words = set(stopwords.words("english"))
    filtered_sents = []
    for sent in tokenized_sents:
        s = [w for w in sent if w not in stop_words]
        filtered_sents.append(s)

    return filtered_sents


def stemming(tokenized_sents):
    # Stemming
    ps = PorterStemmer()

    stemmed_sents = []
    for sent in tokenized_sents:
        s = [ps.stem(w) for w in sent]
        stemmed_sents.append(s)

    return stemmed_sents


def lemmatization(tokenized_sents):
    # lemmatization
    lem = WordNetLemmatizer()
    sents_pos_tag = tag.pos_tag_sents(tokenized_sents, lang="eng")
    wn_sents_pos_tag = []
    for sent in sents_pos_tag:
        wn_sents_pos_tag.append(
            map(lambda tup: (tup[0], convert_nltk2wn_pos(tup[1])), sent)
        )
    lem_sents = []
    for sent in wn_sents_pos_tag:
        s = []
        for word, pos in sent:
            if pos is None:
                s.append(word)
            else:
                s.append(lem.lemmatize(word, pos))
        lem_sents.append(s)
    return lem_sents
