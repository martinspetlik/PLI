import numpy as np
import pandas as pd
from collections import Counter
import re
from language_model import LanguageModel

import fasttext
PRETRAINED_MODEL_PATH = '/tmp/lid.176.bin'
# import fasttext.util
# fasttext.util.download_model('cs', if_exists='ignore')  # Czech
# fasttext.util.download_model('sk', if_exists='ignore')  # Slovak
# ft_cs = fasttext.load_model('cc.cs.300.gz')
# ft_sk = fasttext.load_model('cc.sk.300.gz')


def pli_test():
    #process_corpus()
    train_text, test_text = get_text("czech_corpus.txt")

    czech_lm = LanguageModel(text=train_text.strip())
    czech_lm.run_trigram()
    exit()

    #train_text = "abcdcaadd"
    czech_lm = LanguageModel(text=train_text)
    czech_lm.run_bigram()

    train_text, test_text = get_text("slovak_corpus.txt")

    slovak_lm = LanguageModel(text=train_text)
    slovak_lm.run_bigram()

    #test_text = "Vyšší daně nebudou platit kupříkladu až od května"
    classification(czech_lm, slovak_lm, text=test_text[0:10])

    # tr = LanguageModel()
    # tr.test_trigram()
    #
    model = fasttext.load_model(PRETRAINED_MODEL_PATH)
    sentences = test_text
    predictions = model.predict(sentences)
    print(predictions)


def classification(czech_lm, slovak_lm, text):
    prior_prob = 0.5  # 2 language models are expected
    test_bigram, _ = LanguageModel.create_bigram(text)
    print("test bigram ", test_bigram)

    cs_prob = 0
    sk_prob = 0
    for pair in test_bigram:
        print("pair ", pair)
        ch = pair.split(' ')

        if ch[0] == '':
            ch[0] = ' '

        if ch[1] == '':
            ch[1] = ' '

        # print("ch0: {}, ch1: {}".format(ch0, ch1))
        # print("i0: {} , i1: {}".format(czech_lm.chars_keys[ch0], czech_lm.chars_keys[ch1]))
        # print("")

        if ch[1] not in czech_lm.chars_keys or ch[0] not in czech_lm.chars_keys:
            continue

        prob = np.log(czech_lm.prob_matrix[czech_lm.chars_keys[ch[0]], czech_lm.chars_keys[ch[1]]])
        print("PROB ", prob)
        cs_prob += np.log(czech_lm.prob_matrix[czech_lm.chars_keys[ch[0]], czech_lm.chars_keys[ch[1]]])
        print("cs prob ", cs_prob)
        sk_prob += np.log(slovak_lm.prob_matrix[slovak_lm.chars_keys[pair[0]], slovak_lm.chars_keys[pair[1]]])
        print("sk prob ", sk_prob)

    cs_prob = np.exp(cs_prob) * 0.5 if cs_prob != 0 else 0
    sk_prob = np.exp(sk_prob) * 0.5 if sk_prob != 0 else 0

    print("CS: {}, SK: {}".format(cs_prob, sk_prob))


def get_text(file):
    with open(file, "r") as reader:
        sentences = reader.readlines()

    weights = [0.8, 0.2]
    size = len(sentences)
    mixture_idx = np.random.choice(len(weights), size=size, replace=True, p=weights)

    train = []
    test = []
    for idx, sentence in zip(mixture_idx, sentences):
        if idx == 0:
            train.append(sentence.strip())
        else:
            test.append(sentence.strip())

    train_text = ' '.join(train)
    test_text = ' '.join(test)

    print("train text ", train_text)
    print("text text ", test_text)

    return train_text, test_text


# def process_corpus():
#     with open('corpus.txt', 'r', encoding='utf-8') as reader:  # read from file in utf-16 format
#         corpus_utf16 = reader.readlines()  # read line by line
#
#     czech_sentences = []
#     slovak_sentences = []
#     for line in corpus_utf16:
#         # get the last token of the sentence as the language, the others as the sentence
#
#         sentence, lang = line.rsplit(None, 1)
#         if lang == 'cz':
#             czech_sentences.append(sentence)
#
#         if lang == 'sk':
#             slovak_sentences.append(sentence)
#
#     with open("czech_corpus.txt", "w") as writer:
#         for sentence in czech_sentences:
#             writer.write(sentence + "\n")
#
#     with open("slovak_corpus.txt", "w") as writer:
#         for sentence in slovak_sentences:
#             writer.write(sentence + "\n")
#
#
#     print("len(czech sentences) ", len(czech_sentences))
#     print("czech sentences ", czech_sentences)
#     print("len(slovak sentences) ", len(slovak_sentences))


if __name__ == "__main__":
    pli_test()

    # import cProfile
    # import pstats
    #
    # pr = cProfile.Profile()
    # pr.enable()
    #
    # my_result = pli_test()
    #
    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats('cumtime')
    # ps.print_stats()
