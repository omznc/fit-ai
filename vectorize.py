import os

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Definicije putanja i direktorija
documents_directory = "documents"
chroma_persist_directory = "./chroma"

# Učitavanje potrebnih biblioteka i modela
embeddings = GPT4AllEmbeddings()

print("Starting...")

documents = []
# Koristenje loop-a za iteriranje kroz sve datoteke u direktoriju dokumenata
for filename in os.listdir(documents_directory):
	document = PyPDFLoader(os.path.join(documents_directory, filename)).load_and_split()
	documents.append(document)

if len(documents) == 0:
	print("Dokumenti ne postoje u {}".format(documents_directory))
	exit()

print("Loadano {} dokumenata...".format(len(documents)))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Kombiniranje svih chunk-ova
all_chunks = [text_splitter.split_documents(document) for document in documents]
all_chunks = [chunk for chunks in all_chunks for chunk in chunks]

vector_store = Chroma.from_documents(all_chunks, embeddings, persist_directory=chroma_persist_directory)
vector_store.persist()

print("Done.")
