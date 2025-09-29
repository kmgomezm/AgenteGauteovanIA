# src/nlp_tools.py
from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Tuple, Dict, Any
from pysentimiento import create_analyzer

# ======================
# NER
# ======================
_ner_pipe = None
def ner_pipe():
    global _ner_pipe
    if _ner_pipe is None:
        tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
        model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")

        _ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return _ner_pipe

def _split_text(text: str, max_words: int = 300) -> List[str]:
    """
    Parte el texto en fragmentos de hasta max_words palabras
    para no superar el límite de tokens del modelo.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def ner(text: str) -> List[Tuple[str, str, float]]:
    """
    Ejecuta NER sobre textos largos partiendo en fragmentos.
    Devuelve [(entidad, etiqueta, score)].
    """
    text = (text or "").strip()
    if not text:
        return []

    ents_all = []
    for chunk in _split_text(text, max_words=300):
        ents = ner_pipe()(chunk)
        ents_all.extend([
            (ent["word"], ent["entity_group"], round(float(ent["score"]), 3))
            for ent in ents
        ])
    return ents_all


# ======================
# Sentimiento
# ======================
_sent = None
def sentiment_pipe():
    global _sent
    if _sent is None:
        _sent = create_analyzer(task="sentiment", lang="es")
    return _sent

def sentiment(text: str) -> Dict[str, Any]:
    """
    Analiza sentimiento con pysentimiento.
    Devuelve {'label': etiqueta, 'probs': {'NEG': x, 'NEU': y, 'POS': z}}
    """
    text = (text or "").strip()
    if not text:
        return {"label": "NEU", "probs": {"NEG": 0.0, "NEU": 1.0, "POS": 0.0}}

    try:
        res = sentiment_pipe().predict(text[:1000])  # límite de seguridad
        return {
            "label": res.output,
            "probs": {k: round(v, 3) for k, v in res.probas.items()}
        }
    except Exception as e:
        return {"label": "NEU", "probs": {"NEG": 0.0, "NEU": 1.0, "POS": 0.0}, "error": str(e)}


# ======================
# Zero-shot
# ======================
_zs_pipe = None
def zero_shot_pipe():
    global _zs_pipe
    if _zs_pipe is None:
        _zs_pipe = pipeline(
            "zero-shot-classification",
            model="Recognai/zeroshot_selectra_small"
        )
    return _zs_pipe

def classify(text: str, labels: List[str]) -> List[Tuple[str, float]]:
    text = (text or "").strip()
    if not text:
        return [(label, 0.0) for label in labels]

    try:
        res = zero_shot_pipe()(
            text,
            candidate_labels=labels,
            hypothesis_template="Este texto trata sobre {}.",
            multi_label=True
        )
        pares = list(zip(res["labels"], [round(s, 3) for s in res["scores"]]))
        return sorted(pares, key=lambda x: x[1], reverse=True)
    except Exception as e:
        return [("ERROR", 0.0), (f"{type(e).__name__}: {e}", 0.0)]
