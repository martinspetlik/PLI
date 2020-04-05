from copy import deepcopy
import itertools
import pickle

czech_chars = ['a', 'á', 'b', 'c', 'č', 'd', 'ď', 'e', 'ě', 'é', 'f', 'g', 'h', 'i', 'í', 'j', 'k', 'l', 'm', 'n', 'ň',
               'o', 'ó', 'p', 'q', 'r', 'ř', 's', 'š', 't', 'ť', 'u', 'ú', 'ů', 'v', 'w', 'x', 'y', 'ý', 'z', 'ž']

czech_diacritics_map = {'á': 'a', 'č': 'c', 'ď': 'd', 'ě': 'e', 'é': 'e', 'í': 'i', 'ň': 'n', 'ó': 'o', 'ř': 'r',
                        'š': 's', 'ť': 't', 'ú': 'u', 'ů': 'u', 'ý': 'y', 'ž': 'z'}

slovak_chars = ['a', 'á', 'ä', 'b', 'c', 'č', 'd', 'ď', 'e', 'é', 'f', 'g', 'h', 'i', 'í', 'j', 'k', 'l', 'ĺ', 'ľ',
                'm', 'n', 'ň', 'o', 'ó', 'ô', 'p', 'q', 'r', 'ŕ', 's', 'š', 't', 'ť', 'u', 'ú', 'v', 'w', 'x', 'y',
                'ý', 'z', 'ž']

slovak_diacritics_map = {'á': 'a', 'ä': 'a', 'č': 'c', 'ď': 'd', 'é': 'e', 'í': 'i',  'ĺ': 'l', 'ľ': 'l', 'ň': 'n',
                         'ó': 'o', 'ô': 'o', 'ŕ': 'r', 'š': 's', 'ť': 't', 'ú': 'u', 'ů': 'u', 'ý': 'y', 'ž': 'z'}


all_chars = list(set().union(czech_chars, slovak_chars, "#", ' '))
all_diacritics_map = dict(czech_diacritics_map, **slovak_diacritics_map)
all_chars_without_diacritics = {all_diacritics_map[char] if char in all_diacritics_map else char for char in all_chars}


class LanguageModel:
    def __init__(self, no_diacritics=False):
        possible_chars = all_chars_without_diacritics if no_diacritics else all_chars

        self.trigram_lang_model = None
        self.possible_chars = possible_chars
        self.all_possible_trigrams = {"".join(item) for item in itertools.product(possible_chars, repeat=3)}
        self.trigram_counts = {}
        self.smoothed_trigram = {}

    def update_trigram_counts(self, text):
        sentences = ['##'+s for s in text.split('.')]

        for sentence in sentences:
            for i in range(len(sentence) - 2):
                chars = sentence[i:i + 3]

                if chars in self.all_possible_trigrams:
                    if chars in self.trigram_counts:
                        self.trigram_counts[chars] += 1
                    else:
                        preceding_chars = chars[0:2]
                        # Add other variations
                        for char in self.possible_chars:
                            new_trigram = preceding_chars + char
                            if new_trigram not in self.trigram_counts:
                                self.trigram_counts[new_trigram] = 0
                        self.trigram_counts[chars] = 1

        counts_sum = sum(self.trigram_counts.values())
        self.trigram_lang_model = {chars: count / counts_sum for chars, count in self.trigram_counts.items()}

        self.witten_bell_discounting()

    def witten_bell_discounting(self):
        smoothed_trigram = deepcopy(self.trigram_lang_model)
        counted_N_T_Z = {}

        for chars, count in self.trigram_counts.items():
            N = 0  # the number of occurrences of the preceding chars
            T = 0  # all triples with preceding chars (from given data)
            Z = 0  # all possible pairs with preceding chars
            preceding_chars = chars[:2]

            if preceding_chars in counted_N_T_Z:
                N, T, Z = counted_N_T_Z[preceding_chars]
            else:
                for ch, c in self.trigram_counts.items():
                    if ch.startswith(preceding_chars):
                        N += c
                        if c > 0:
                            T += 1
                        if c == 0:
                            Z += 1
                counted_N_T_Z[preceding_chars] = N, T, Z

            if count == 0:
                smoothed_trigram[chars] = T / (Z * (N + T))
            else:
                smoothed_trigram[chars] = count / (N + T)

        self.smoothed_trigram = smoothed_trigram


def save_language_model(l_model, path):
    with open(path, "wb") as writer:
        pickle.dump(l_model, writer)


def load_language_model(path):
    with open(path, "rb") as writer:
        l_model = pickle.load(writer)

    return l_model
