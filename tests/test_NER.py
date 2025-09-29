from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")

print("Tokenizer max length:", tokenizer.model_max_length)
print("Model max embeddings:", model.config.max_position_embeddings)

pipe = pipeline("ner", 
                model="Babelscape/wikineural-multilingual-ner",
                grouped_entities=True  # agrupa entidades
                )

text = "afecto entre compañeros, la solidaridad, incluso la competencia, habilidades sociales complementarias en una educación integra. Y que decir de la sensibilidad ecológica que se construía en las excursiones por regiones de los países, procurando el contacto con diversos fenómenos geográficos y naturales. El papel del arte en el fortalecimiento de la inteligencia emocional y el deporte como crisol de otros modos de relaciones sociales que significan la existencia y calidad de la especie. Poco de todo esto lo alcanza a suplir la clase virtual, que por su parte aporta otros lenguajes, , otras libertades y sobre todo múltiples recursos que agilizan la adquisición de datos y el logro de sistemas y maquetas de modo tridimensional si es preciso. Es una encrucijada interesante para la nueva humanidad. Me pregunto por tipo de idoneidad que asegurará el profesionalismo de arquitectos e ingenieros de ahora. De hecho las maquetas hace rato que son juguetes rupestres que no usa nadie, si los programas de computador cuentan con el modo de reproducir en todas las dimensiones espacios y sistemas por ellos creados. Pero el contacto con los materiales. La valoración de los ecosistemas en cada lote a construir siempre exige un contacto con el material, que no lo suple la información topográfica que da una GPS. Ya no hará falta untarse del barro del terreno como nos lo enseño el maestro Salmona en su momento. No crítico los caminos que han procurado las instituciones, al contrario confío en que la pedagogía siempre sabrá desempeñarse con conciencia del presente. Se trata de no dejarle todo a las empresas de tecnología digital para que lo humano se libere de quedar supeditado a la máquina. Grande es el reto"
ents = pipe(text)

print("Salida cruda del pipeline:")
for e in ents:
    print(e)

