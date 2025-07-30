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
    return text


print(clean_text("Anmum Emesa Chocolate 200Gr – Susu Bubuk, Menyusui! Berkualitas."))
