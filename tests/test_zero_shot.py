# tests/test_zero_shot.py
from src import nlp_tools

print("=== Test de Zero-Shot ===")

# Etiquetas de prueba
labels = ["política", "salud", "ciencia", "economía", "educación"]

# Algunos textos de prueba
textos = [
    "El Congreso aprobó una nueva ley de educación en 2019.",
    "La pandemia impactó en el sistema de salud y la economía.",
    "El telescopio espacial descubrió un nuevo planeta.",
    "",
    "x" * 5000,  # texto muy largo para forzar truncado
]

for t in textos:
    print("\nTexto:", repr(t[:60]) + ("..." if len(t) > 60 else ""))
    try:
        res = nlp_tools.classify(t, labels)
        print("Salida (label, score):")
        for label, score in res:
            print(f"  - {label}: {score}")
    except Exception as e:
        print("⚠️ Error:", type(e).__name__, str(e))
