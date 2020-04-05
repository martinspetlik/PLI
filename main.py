import os
import numpy as np
from language_model import LanguageModel, save_language_model, load_language_model
from data_preparation import prepare_data, format_text, remove_diacritics
from results_export import PDFResults

import fasttext
FASTTEXT_MODEL_PATH = 'lid.176.bin'

DATA_PATH = "data/{}"
LANG_MODELS = "lang_models"

CS_MODEL_DIACRITICS = "cs_model_diacritics"
CS_MODEL = "cs_model"
SK_MODEL_DIACRITICS = "sk_model_diacritics"
SK_MODEL = "sk_model"


def create_trigram(sk=False, no_diacritics=False):
    if sk:
        dir_path = DATA_PATH.format('slovak')
    else:
        dir_path = DATA_PATH.format('czech')

    lm_model = LanguageModel(no_diacritics=no_diacritics)
    files = prepare_data(dir_path, no_diacritics=no_diacritics)

    for file in files:
        with open(file, "r") as reader:
            text = reader.read()
            lm_model.update_trigram_counts(text)

    return lm_model


def create_models():
    cs_model = create_trigram(sk=False, no_diacritics=False)
    save_language_model(cs_model, os.path.join(LANG_MODELS, CS_MODEL))

    sk_model = create_trigram(sk=True, no_diacritics=False)
    save_language_model(sk_model, os.path.join(LANG_MODELS, SK_MODEL))

    cs_model_no_diacritics = create_trigram(sk=False, no_diacritics=True)
    save_language_model(cs_model_no_diacritics, os.path.join(LANG_MODELS, CS_MODEL_DIACRITICS))

    sk_model_no_diacritics = create_trigram(sk=True, no_diacritics=True)
    save_language_model(sk_model_no_diacritics, os.path.join(LANG_MODELS, SK_MODEL_DIACRITICS))


def classification(cs_model, cs_model_no_dicritics, sk_model, sk_model_no_diacritics, text, no_diacritics=False):
    sentences = ['##' + sentence for sentence in
                 text.split('.')]

    sk_prob, cz_prob = 0, 0
    for sentence in sentences:
        for i in range(len(sentence) - 2):
            try:
                if no_diacritics:
                    sk_prob += np.log(sk_model_no_diacritics.smoothed_trigram[sentence[i:i + 3]])
                    cz_prob += np.log(cs_model_no_dicritics.smoothed_trigram[sentence[i:i + 3]])
                else:
                    sk_prob += np.log(sk_model.smoothed_trigram[sentence[i:i + 3]])
                    cz_prob += np.log(cs_model.smoothed_trigram[sentence[i:i + 3]])
            except:
                 Exception("Unknown trigram")

    return "cs" if cz_prob > sk_prob else "sk" if cz_prob < sk_prob else "cs=sk"


def test_models():
    cs_model, cs_model_no_diacritics, sk_model, sk_model_no_diacritics = None, None, None, None
    fasttext_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

    pdf_result = PDFResults()

    sk_model_no_diacritics = load_language_model(os.path.join(LANG_MODELS, SK_MODEL_DIACRITICS))
    sk_model = load_language_model(os.path.join(LANG_MODELS, SK_MODEL))
    cs_model_no_diacritics = load_language_model(os.path.join(LANG_MODELS, CS_MODEL_DIACRITICS))
    cs_model = load_language_model(os.path.join(LANG_MODELS, CS_MODEL))


    def test_cases(no_diacritics=False):

        sentences = {"\"Každopádně teď nechystáme nějaké zásadní změny, musíme to ještě chvilku vydržet\" uvedl po schůzce s"
                     " prezidentem Zemanem vicepremiér Jan Hamáček (ČSSD).": "cs",
                     "Stát v březnu oznámil, že lidem uvolněným z práce kvůli pandemii koronaviru vyplatí 80 procent jejich"
                     " mzdy do maximální výše 2500 liber.": "cs",
                     "Britský výzkum ale ukazuje, že užívání infotainmentu zkracuje reakční dobu řidiče mnohem víc než některé"
                     " omamné látky.": "cs",
                     "Zmeny v Zákonníku práce, ktoré tento týždeň schválil parlament, podľa odborníkov pomôžu zmierniť následky"
                     " krízy spôsobenej novým koronavírusom.": "sk",
                     "Je namieste, aby sa zabránilo prepadu životnej úrovne, a tým aj dopytu v ekonomike. Netreba však zabúdať,"
                     " že síce príslušný fond je pravidelne prebytkový, avšak to bude mať priamy negatívny vplyv na"
                     " hospodárenie Sociálnej poisťovne (SP) ako celku": "sk",
                     "Následky dnešného porušovania opatrení tak uvidíme o mesiac, hovorí popredná epidemiologička Zuzana"
                     " Krištúfková zo Slovenskej zdravotníckej univerzity.": "sk"
                     }

        #######################
        ### Jednotlivé věty ###
        #######################
        pdf_result.append("Jednotlivé věty", align="C")
        for sentence, lang in sentences.items():
            if no_diacritics:
                pdf_result.append(remove_diacritics(sentence))
            else:
                pdf_result.append(sentence)
            sentence = format_text(sentence, no_diacritics=no_diacritics)
            res = classification(cs_model, cs_model_no_diacritics, sk_model, sk_model_no_diacritics, sentence)

            pdf_result.append_results("očekávaný jazyk: {}, výsledek clasifikace: {}".format(lang, res))
            pdf_result.append_results("fasttext výsledek: {}".format(fasttext_model.predict(format_text(sentence, no_diacritics=no_diacritics))))
            pdf_result.add_empty_line()

        ##########################
        ###        Knihy       ###
        ##########################
        pdf_result.append("Celé texty", align="C")
        with open('maly_princ_cs.txt', 'r', encoding='UTF-8') as reader:
            cs_maly_princ = format_text(reader.read(), no_diacritics=no_diacritics)

        #### Maly princ česky
        pdf_result.append("kniha Malý princ - česky")

        res = classification(cs_model, cs_model_no_diacritics, sk_model, sk_model_no_diacritics, cs_maly_princ,
                             no_diacritics=no_diacritics)

        pdf_result.append_results("očekávaný jazyk: {}, výsledek klasifikace: {}".format("cs", res))
        pdf_result.append_results("fasttext výsledek: {}".format(fasttext_model.predict(format_text(cs_maly_princ,
                                                                                                    no_diacritics=no_diacritics))))
        ### Maly princ slovensky
        pdf_result.append("kniha Malý princ - slovensky")
        with open('maly_princ_sk.txt', 'r', encoding='UTF-8') as reader:
            sk_maly_princ = format_text(reader.read(), no_diacritics=no_diacritics)

        res = classification(cs_model, cs_model_no_diacritics, sk_model, sk_model_no_diacritics, sk_maly_princ,
                             no_diacritics=no_diacritics)

        pdf_result.append_results("očekávaný jazyk: {}, výsledek klasifikace: {}".format("sk", res))


        res = classification(cs_model, cs_model_no_diacritics, sk_model, sk_model_no_diacritics, sk_maly_princ,
                             no_diacritics=no_diacritics)

        print("expected: {}, result: {}".format("sk", res))
        pdf_result.append_results("fasttext výsledek: {}".format(fasttext_model.predict(format_text(sk_maly_princ,
                                                                                                    no_diacritics=no_diacritics))))

        ##########################################
        ###    Knihy po jednotlivých větách    ###
        ##########################################
        pdf_result.append("kniha Malý princ - česky, úspěšnost po jednotlivých větách")
        results = {"cs": 0, "sk": 0}
        for sentence in ["##" + sentence for sentence in cs_maly_princ.split('.')]:
            res = classification(cs_model, cs_model_no_diacritics, sk_model, sk_model_no_diacritics, sentence,
                                 no_diacritics=no_diacritics)
            results[res] += 1

        pdf_result.append_results("cs: {}/{}, sk: {}/{}".format(results["cs"], results["cs"] + results["sk"],
                                                       results["sk"], results["cs"] + results["sk"]))


        pdf_result.append("kniha Malý princ - slovensky, úspěšnost po jednotlivých větách")
        results = {"cs": 0, "sk": 0}
        for sentence in ["##" + sentence for sentence in sk_maly_princ.split('.')]:
            res = classification(cs_model, cs_model_no_diacritics, sk_model, sk_model_no_diacritics, sentence,
                                 no_diacritics=no_diacritics)
            results[res] += 1
        pdf_result.append_results("cs: {}/{}, sk: {}/{}".format(results["cs"], results["cs"] + results["sk"],
                                                                results["sk"], results["cs"] + results["sk"]))


    #############################
    ### Texty s diakritikou   ###
    #############################
    pdf_result.append_heading("Texty s diakritikou")

    test_cases(no_diacritics=False)

    #############################
    ### Texty bez diakritiky  ###
    #############################
    pdf_result.append_heading("Texty bez diakritiky")

    test_cases(no_diacritics=True)
    pdf_result.print()

if __name__ == "__main__":
    create_models()
    test_models()
