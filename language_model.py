import numpy as np
import pandas as pd
from collections import Counter
import re


class LanguageModel:
    def __init__(self, text):
        self.text = text
        self.trigram = []
        self.bigram = []

        self.prob_trigram = dict()
        self.prob_bigram = dict()
        self.prob_matrix = None

        self.chars_keys = {}

    def run_bigram(self):
        # Text to letters
        characters = list(self.text)

        self._create_bigram(characters)
        unique_chars = set(characters)
        self._conditional_prob_bigram()

        probs = np.zeros((len(unique_chars), len(unique_chars)))
        print("probs ", probs)

        unique_chars = sorted(list(set(characters)))#['s', 'a', 'b', 'c', 'd']
        print("unique chars ", unique_chars)
        self.unique_chars = unique_chars

        for i1, ch1 in enumerate(unique_chars):
            for i2, ch2 in enumerate(unique_chars):
                probs[i2, i1] = self.prob_bigram.get(ch1 + "|" + ch2, 0)
                print(ch1 + "|" + ch2 + " = " + str(probs[i1, i2]))

        # print("final probs ")
        # print(pd.DataFrame(probs))

        self.prob_matrix = self._witten_bell_bigram()

    def _create_bigram(self, data):
        """
        Create bigram -> List of pairs of following letters
        :param data: list of letters
        :return: None
        """
        self.bigram, self.unigram = LanguageModel.create_bigram(data)

    def _conditional_prob_bigram(self):
        """
        Create bigram conditional probabilities
        :return:
        """
        bigrams_count = Counter(self.bigram)
        unigrams_count = Counter(self.unigram[:-1])

        for i in range(len(self.bigram)):
            b = self.bigram[i].split(' ')
            key = b[1] + '|' + b[0]

            if b[0] == '':
                b[0] = ' '

            value = (bigrams_count[self.bigram[i]]) / (unigrams_count[b[0]])
            self.prob_bigram[key] = value

        self._calc_entropy()

    def _calc_entropy(self):
        entropy = 0
        for condition, prob in self.prob_bigram.items():
            entropy += (prob * np.log2(prob))

        entropy = entropy * (-1)
        print("entropy ", entropy)

    def _witten_bell_bigram(self):
        """
        Bigram Witten-Bell discounting
        :return:
        """
        unigram_count = Counter(self.unigram[:-1])
        bigram_count = Counter(self.bigram)
        print("unigram count ", unigram_count)
        print("bigram count ", bigram_count)

        unique_chars = self.unique_chars
        probs = np.zeros((len(unique_chars), len(unique_chars)))

        for i1, ch1 in enumerate(unique_chars):
            for i2, ch2 in enumerate(unique_chars):
                if ch1 not in self.chars_keys:
                    self.chars_keys[ch1] = i1
                ngram_count = bigram_count.get(ch2 + ' ' + ch1, 0)
                #print("c({}{}) = {}".format(ch2, ch1, ngram_count))

                N = unigram_count.get(ch2, 0)
                # N - the number of occurrences of the letter 'ch2' as a precursor

                T = 0
                for key, value in bigram_count.items():
                    #print("ch2 ", ch2)
                    if re.findall(rf'^{re.escape(ch2)}.*', key):
                        T += 1
                # T - all pairs with "ch2" as a precursor (from given data)

                Z = len(unique_chars) - T
                # Z - all possible pairs with "ch2" as precursor
                if ngram_count == 0:
                    prob = float(T) / float(Z * (N + T))
                else:
                    prob = float(ngram_count) / float(N + T)

                # print("{} | {}".format(ch1, ch2))
                # print("c({}{}) = {}".format(ch2, ch1, ngram_count))
                # print("N = {}, T = {}, Z = {}".format(N, T, Z))
                # print("prob ", prob)

                probs[i2, i1] = prob


        print("probs ")
        print(pd.DataFrame(probs))
        print("np.sum(probs, axis=1) ", np.sum(probs, axis=1))

        assert np.allclose(np.sum(probs, axis=1), np.ones(len(unique_chars)))

        return probs

    def run_trigram(self):
        characters = list(self.text)

        self._create_bigram(characters)
        self._conditional_prob_bigram()

        self._create_trigram(characters)
        self._conditional_prob_trigram()

        unique_chars = set(characters)
        unique_chars = sorted(list(set(characters)))
        self.unique_chars = unique_chars

        probs = np.zeros((len(unique_chars), len(unique_chars), len(unique_chars)))
        print("probs ", probs)

        # for i1, ch1 in enumerate(unique_chars):
        #     for i2, ch2 in enumerate(unique_chars):
        #         for i3, ch3 in enumerate(unique_chars):
        #             probs[i1, i2, i3] = self.prob_trigram.get(ch3 + '|' + ch1 + ' ' + ch2, 0)
        #
        #             print(ch1 + "|" + ch2 + " = " + str(probs[i1, i2]))

        # print("final probs ")
        # print(probs)

        self._witten_bell_trigram()

        print("prob trigram ", self.prob_trigram)

    def _create_trigram(self, data):
        self.trigram, self.unigram = LanguageModel.create_trigram(data)
        # self.unigram = data
        # for i in range(len(self.unigram) - 2):
        #     self.trigram.append(self.unigram[i] + ' ' + self.unigram[i + 1] + ' ' + self.unigram[i + 2])

    def _conditional_prob_trigram(self):
        trigrams_count = Counter(self.trigram)
        bigrams_count = Counter(self.bigram)

        print("self bigram ", self.bigram)
        #print("bigrams count ", bigrams_count)
        print("self. trigram ", self.trigram)

        for i in range(len(self.trigram)):
            b = self.trigram[i].split('^')
            key = b[2] + '|' + b[0] + ' ' + b[1]

            # print("b ", b)
            # print("len(b) ", len(b))
            #
            # if b[0] == '':
            #     b[0] = ' '
            # if b[1] == '':
            #     b[1] = ' '
            # if b[2] == '':
            #     b[2] = ' '
            #
            # print("b[0] + ' ' + b[1] ", b[0] + ' ' + b[1])
            # print("bigrams_count", bigrams_count[b[0] + ' ' + b[1]])

            value = (trigrams_count[self.trigram[i]]) / (bigrams_count[b[0] + ' ' + b[1]])
            self.prob_trigram[key] = value


    def _witten_bell_trigram(self):
        """
        Trigram Witten-Bell discounting
        :return:
        """
        bigram_count = Counter(self.bigram)
        trigram_count = Counter(self.trigram)
        print("bigram count ", bigram_count)
        print("trigram count ", trigram_count)

        unique_chars = self.unique_chars
        probs = np.zeros((len(unique_chars), len(unique_chars), len(unique_chars)))

        for i1, ch1 in enumerate(unique_chars):
            for i2, ch2 in enumerate(unique_chars):
                for i3, ch3 in enumerate(unique_chars):
                    ngram_count = trigram_count.get(ch3 + '^' + ch2 + '^' + ch1, 0)
                    print("c({}{}{}) = {}".format(ch3, ch2, ch1, ngram_count))

                    N = bigram_count.get(ch3 + ' ' + ch2, 0)
                    print("bigram count ", bigram_count)
                    print("N = {}".format(N))

                    T = 0
                    for key, value in trigram_count.items():
                        print("key: {}, value: {}".format(key, value))

                        if re.findall(rf'^{ch3}\^{ch2}.*', key):
                            T += 1

                    print("T = {}".format(T))
                    
                    Z = len(self.trigram) - T
                    if ngram_count == 0:
                        prob = float(T) / float(Z * (N + T))
                    else:
                        prob = float(ngram_count) / float(N + T)

                    print("{} | {}{}".format(ch1, ch2, ch3))
                    print("c({}{}{}) = {}".format(ch3, ch2, ch1, ngram_count))
                    print("N = {}, T = {}, Z = {}".format(N, T, Z))
                    print("prob ", prob)

                    probs[i2, i1, i3] = prob

        print("probs ")
        print(probs)
        self.prob_matrix = probs

    @staticmethod
    def create_bigram(data):
        """
        Create bigram -> List of pairs of following letters
        :param data: list of letters
        :return: None
        """
        unigram = data
        bigram = []
        for i in range(len(unigram) - 1):
            bigram.append(unigram[i] + ' ' + unigram[i + 1])
        return bigram, unigram

    @staticmethod
    def create_trigram(data):
        unigram = data
        trigram = []
        for i in range(len(unigram) - 2):
            trigram.append(unigram[i] + '^' + unigram[i + 1] + '^' + unigram[i + 2])

        return trigram, unigram
