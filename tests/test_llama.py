(
echo from ollama import chat
echo resp = chat(model='llama3.1:8b', messages=[{'role':'user','content':'Hola Llama 3.1, listo para trabajar localmente?'}])
echo print(resp['message']['content'])
)