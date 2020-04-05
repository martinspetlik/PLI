import os
import glob
import re
from language_model import all_diacritics_map
PREPARED_DATA = "prep"
PREPARED_DATA_NO_DIACRITICS = "prep_no_diacritics"


def prepare_data(dir_path, no_diacritics=False):

    if no_diacritics:
        prep_path = os.path.join(dir_path, PREPARED_DATA_NO_DIACRITICS)
    else:
        prep_path = os.path.join(dir_path, PREPARED_DATA)

    if not os.path.isdir(prep_path):
        os.mkdir(prep_path)
        files = glob.glob(os.path.join(dir_path, "wiki_*"))

        for file in files:
            with open(file, "r") as reader:
                text = format_text(reader.read(), no_diacritics)

            file_name = os.path.basename(file)
            with open(os.path.join(prep_path, file_name), "w") as writer:
                writer.write(text)

    return glob.glob(os.path.join(prep_path, "wiki_*"))


def format_text(text, no_diacritics):
    text = text.lower()

    text = re.sub(re.compile('<.*?>'), '', text)  # remove tags
    text = re.sub(re.compile('[^a-zá-ž. äľôŕ]'), ' ', text)  # remove keep alphabetic
    text = re.sub(re.compile(' +'), ' ', text)  # remove multiple spaces
    text = re.sub(re.compile('\. '), '.', text)  # remove space after dot

    if no_diacritics:
        text = remove_diacritics(text)
    return text


def remove_diacritics(text):
    return text.translate(str.maketrans("".join(all_diacritics_map.keys()), "".join(all_diacritics_map.values())))
