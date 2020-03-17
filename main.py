import numpy as np
import pandas as pd
import nltk
#nltk.download()
from collections import Counter
from nltk import trigrams, ngrams
import random


class Trigram:
    def __init__(self):
        self.trigram = []
        self.bigram = []

        self.prob_trigram = dict()
        self.prob_bigram = dict()

    def create(self):

        bigram_text = "<s> dnes respektive dnes možná se uvidíme v kině"
        text = "<s> <s> dnes respektive dnes možná se uvidíme v kině"

        print("text.split() ", text.split())
        characters = list(text)
        characters = list("sabcdcaadd")
        #print("characters ", characters)

        ### Text trigram ###
        # token = nltk.word_tokenize(text)
        # trigram = ngrams(token, 3)
        # print("NLTK trigarms ", trigram)


        # self._create_bigram(text.split())
        # self._create_trigram(text.split())

        self._create_bigram(characters)

        print("set characters ", set(characters))
        # self._create_trigram(text.split())

        unique_chars = set(characters)

        # print("self.trigram ", self.trigram)
        # print("Counter(self.trigram) ", Counter(self.trigram))

        self._conditional_prob_bigram()

        probs = np.zeros((len(unique_chars), len(unique_chars)))
        print("probs ", probs)

        unique_chars = ['s', 'a', 'b', 'c', 'd']

        for i1, ch1 in enumerate(unique_chars):
            for i2, ch2 in enumerate(unique_chars):
                probs[i1, i2] = self.prob_bigram.get(ch1 + "|" + ch2, 0)

                print(ch1 + "|" + ch2 + " = " + str(probs[i1, i2]))

        print("final probs ")
        print(pd.DataFrame(probs))

        self._witten_bell_smoothing(probs)



        print("prob trigram ", self.prob_trigram)

        # Characters trigram
        # token = nltk.word_tokenize(characters)
        # trigram = ngrams(token, 3)
        # print("NLTK trigarms ", trigram)

        # self._create_trigram(characters)
        # print("self.trigram ", self.trigram)
        # print("Counter(self.trigram) ", Counter(self.trigram))

    def _witten_bell_smoothing(self, probs):
        pass

    def _create_trigram(self, data):
        self.unigrams = data
        for i in range(len(self.unigrams) - 2):
            self.trigram.append(self.unigrams[i] + ' ' + self.unigrams[i + 1] + ' ' + self.unigrams[i + 2])

    def _create_bigram(self, data):
        self.unigrams = data
        for i in range(len(self.unigrams) - 1):
            self.bigram.append(self.unigrams[i] + ' ' + self.unigrams[i + 1])

    def _conditional_prob_bigram(self):
        bigrams_count = Counter(self.bigram)
        print("bigram count ", bigrams_count)
        unigrams_count = Counter(self.unigrams[:-1])
        for i in range(len(self.bigram)):
            b = self.bigram[i].split()
            key = b[1] + '|' + b[0]

            print("key ", key)
            print("bigrams_count[self.bigram[i]] ", bigrams_count[self.bigram[i]])
            print("unigrams_count[b[0]] ", unigrams_count[b[0]])

            value = (bigrams_count[self.bigram[i]]) / (unigrams_count[b[0]])

            self.prob_bigram[key] = value
        print("prob_bigram ", self.prob_bigram)

    def _conditional_prob_trigram(self):
        trigrams_count = Counter(self.trigram)
        bigrams_count = Counter(self.bigram)

        for i in range(len(self.trigram)):
            b = self.trigram[i].split()
            key = b[2] + '|' + b[0] + ' ' + b[1]
            value = (trigrams_count[self.trigram[i]]) / (bigrams_count[b[0] + ' ' + b[1]])
            self.prob_trigram[key] = value


if __name__ == "__main__":
    tr = Trigram()
    tr.create()
