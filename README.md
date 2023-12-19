# FIT projekt iz umjetne inteligencije

## Opis projekta

Koristenje Chroma db-a za pohranjivanje splitovanih chunkova PDF-ova koji se mogu pretraživati po ključnim riječima.

Te pretrage mozemo dati LLM-u, u ovom slucaju GPT4ALL ili OpenAI-kompatibilnom API-u kako bi smo imali chat-with-PDF sistem.


## Koristenje

Python 3.10 je potreban, 3.11 ima problema sa langchain-om.

```bash
pip install -r requirements.txt
```

### OpenAI compatible
    
Potrebno je modify-ati `query-openai-compat.py` sa API key-em i imenom modela, ili po defaultu koristiti LM Studio.

Ako koristimo LM Studio, u testiranju je korišten `TheBloke/Mistral-7B-Instruct-v0.2-GGUF` model.

https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
```bash
python vectorize-openai-compat.py # Napraviti ce chroma-openai-compat folder 
python query-openai-compat.py
```

### GPT4ALL

Potrebno je preuzeti neki LLM, u ovom slučaju je korišten orca-mini-3b, i staviti ga u `models/` folder.

https://huggingface.co/pankajmathur/orca_mini_3b
```bash
python vectorize.py # Napraviti ce chroma folder
python query.py
```
