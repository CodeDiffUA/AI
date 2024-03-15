import pandas as pd
import stanza
import nltk
import re

#stanza.download('uk')
nlp = stanza.Pipeline('uk')


def lemmatize(sentence):
    doc = nlp(sentence)
    lemmatized_words = [word.lemma for sent in doc.sentences for word in sent.words]
    return ' '.join([word.strip().lower() for word in lemmatized_words if word.strip()])


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [lemmatize(word) for word in tokenized_sentence]
    bag = [0] * len(all_words)
    for s in tokenized_sentence:
        for i, word in enumerate(all_words):
            if word == s:
                bag[i] = 1
    return bag


def clean_text(x):
    pattern = r'[ˆa-zA-z0-9\s]'
    x = re.sub(pattern, ' ', x)
    return x
