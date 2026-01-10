import re
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import json
import os

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("indonesian"))
stemmer = StemmerFactory().create_stemmer()

def load_slang():
    if os.path.exists("slangwords.json"):
        with open("slangwords.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

slang_dict = load_slang()

def preprocessing_pipeline(text: str) -> dict:
    # Case folding
    case_folding = text.lower()

    # Cleaning
    cleaning = unicodedata.normalize("NFKD", case_folding)
    cleaning = re.sub(r"http\S+|www\S+", " ", cleaning)
    cleaning = re.sub(r"\d+", " ", cleaning)
    cleaning = re.sub(r"[^a-z\s]", " ", cleaning)
    cleaning = re.sub(r"\s+", " ", cleaning).strip()

    # Tokenizing
    tokenizing = word_tokenize(cleaning)

    # Normalisasi
    normalisasi = [slang_dict.get(w, w) for w in tokenizing]

    # Stopword removal
    stopword_removal = [w for w in normalisasi if w not in stop_words]

    # Stemming
    stemming = [stemmer.stem(w) for w in stopword_removal]

    # Final text
    final_text = " ".join(stemming)

    return {
        "case_folding": case_folding,
        "cleaning": cleaning,
        "tokenizing": tokenizing,
        "normalisasi": normalisasi,
        "stopword_removal": stopword_removal,
        "stemming": stemming,
        "final_text": final_text
    }
