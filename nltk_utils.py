import stanza
import nltk
import numpy as np
import logging
import os
os.environ["STANZA_RESOURCES"] = "/dev/null"
logging.basicConfig(level=logging.ERROR)

# stanza.download('uk')
nlp = stanza.Pipeline('uk', logging_level='ERROR')


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def lemmatize(sentence):
    doc = nlp(sentence)
    lemmatized_words = [word.lemma for sent in doc.sentences for word in sent.words]
    return ' '.join([word.strip().lower() for word in lemmatized_words if word.strip()])


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [lemmatize(word) for word in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w, in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag
