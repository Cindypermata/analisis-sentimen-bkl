
from rapidfuzz.distance import DamerauLevenshtein
import re
import os


def load_kamus(path="kamus.txt"):
    if not os.path.exists(path):
        # Streamlit-friendly: diamkan saja
        return set()

    vocab = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            kata = line.strip().lower()
            if kata:
                vocab.add(kata)

    return vocab


kamus_baku = load_kamus()

# NORMALISASI HURUF BERULANG
def remove_repeated_letters(word):
    # bangeeet → banget
    return re.sub(r"(.)\1{1,}", r"\1", word)


# KOREKSI KATA
def correct_word_distance(word, vocab, max_distance=0.25):
    word = word.lower()
    word_clean = remove_repeated_letters(word)

    if word_clean in vocab:
        return word_clean

    best_word = word_clean
    best_dist = 1.0

    for v in vocab:
        dist = DamerauLevenshtein.normalized_distance(word_clean, v)
        if dist < best_dist:
            best_dist = dist
            best_word = v

    if best_dist > max_distance:
        return word_clean

    return best_word


# KOREKSI TOKEN (LIST)
def dld_correct_tokens(tokens, vocab=kamus_baku, max_distance=0.25):
    return [
        correct_word_distance(word, vocab, max_distance)
        for word in tokens
    ]

# KOREKSI TEKS UTUH (STRING)
def dld_correct_text(text, vocab=kamus_baku, max_distance=0.25):
    """
    Input  : string (hasil preprocessing)
    Output : string (hasil DLD)
    """
    if not text or not isinstance(text, str):
        return ""

    tokens = text.split()
    corrected_tokens = dld_correct_tokens(tokens, vocab, max_distance)
    return " ".join(corrected_tokens)
