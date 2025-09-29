# test_sentimiento.py
from transformers import pipeline

print("=== Test de pipeline de sentimiento ===")

MODEL = "pysentimiento/robertuito-sentiment-analysis"

try:
    sent_pipe = pipeline("text-classification", model=MODEL)
    textos = [
        "Me gusta mucho la ciencia de datos.",
        "Este gobierno es un desastre.",
        "Hoy es un día normal.",
        "",
        "x" * 5000,  # texto largo
    ]

    for t in textos:
        print("\nTexto:", repr(t[:60]) + ("..." if len(t) > 60 else ""))
        try:
            res = sent_pipe(t, truncation=True, max_length=256)
            print("Resultado:", res)
        except Exception as e:
            print("⚠️ Error:", type(e).__name__, str(e))

except Exception as e:
    print("❌ No se pudo cargar el modelo:", type(e).__name__, str(e))
