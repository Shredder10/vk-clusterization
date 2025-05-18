import nltk
from pymystem3 import Mystem

def delete_non_letters(words):
    new_words = []
    for word in words:
        new_word = "".join(c for c in word if c.isalpha())
        if new_word != '':
            new_words.append(new_word)
    return new_words

def delete_stopwords(words):
    with open('./python/Data/stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = f.read()
    with open('./python/Data/stopwords_add.txt', 'r', encoding='utf-8') as f:
        stopwords1 = f.read()
    stopwords = stopwords.split(",")
    stopwords1 = stopwords1.split(",")
    stopwords = stopwords + stopwords1
    new_words = []
    for word in words:
        if word not in stopwords:
            new_words.append(word)
    return new_words

def to_lowercase(words):
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def normalize(text, del_stopwords=True):
    mystem = Mystem()
    words = mystem.lemmatize(text)
    text = ''
    for i in range(0, len(words)):
        text = text + words[i]
    text = nltk.word_tokenize(text)
    text = delete_non_letters(text)
    text = to_lowercase(text)

    if del_stopwords:
        text = delete_stopwords(text)
    text = [word for word in text if len(word) > 1]
    return text
