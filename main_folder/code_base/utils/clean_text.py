import re
import string
from unidecode import unidecode
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english")) | set(stopwords.words("indonesian"))


def clean_text(text):
    text = unidecode(text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = word_tokenize(text)
    text = [w for w in text if w not in stop_words and len(w) > 1]
    return " ".join(text)


# print(clean_text("Anmum Emesa Chocolate 200 Gr – Susu Bubuk, Menyusui! Berkualitas."))
