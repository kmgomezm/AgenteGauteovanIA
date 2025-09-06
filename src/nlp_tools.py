import spacy
from transformers import pipeline

_nlp = None
def spacy_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("es_core_news_lg")
    return _nlp

_sent = None
def sentiment_pipe():
    global _sent
    if _sent is None:
        _sent = pipeline("text-classification",
                         model="finiteautomata/beto-sentiment-analysis",
                         top_k=None, truncation=True)
    return _sent

def ner(text):
    doc = spacy_nlp()(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def sentiment(text):
    res = sentiment_pipe()(text[:4000])
    return res

_zs = None
def zero_shot():
    global _zs
    if _zs is None:
        _zs = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")
    return _zs

def classify(text, labels):
    return zero_shot()(text[:800], candidate_labels=labels, hypothesis_template="El texto trata sobre {}.")
