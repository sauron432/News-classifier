import spacy
import pandas as pd

def preprocess(text):
    nlp = spacy.load('en_core_web_lg')
    doc = nlp(text)
    return " ".join(
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct
    )