from langchain.chains import LLMChain
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import GPT4All
from langchain.vectorstores.chroma import Chroma
from langchain_core.prompts import PromptTemplate

PROMPT_TEMPLATE = """
Odgovori samo na pitanja koja se odnose na tekst koji je dat ispod.

{context}

---

Odgovori na ovo pitanje u vezi sa tekstom iznad, na hrvatskom/bosanskom jeziku: {question}
Ne koristi engleski jezik.
"""


def main():
	# Load the DB
	function = GPT4AllEmbeddings()
	db = Chroma(persist_directory="chroma", embedding_function=function)

	model = GPT4All(model="./models/orca-mini-3b-gguf2-q4_0.gguf", device="gpu", n_threads=16) # device/n_threads se mogu promijeniti

	while True:
		query_text = input("Pitanje: ")

		results = db.similarity_search_with_relevance_scores(query_text, k=6)
		if len(results) == 0:
			print(f"Nista nije nadjeno za upit: {query_text}")
			continue

		context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
		prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

		llm_chain = LLMChain(prompt=prompt, llm=model)

		resp = llm_chain.run(context=context_text, question=query_text, max_tokens=50)

		sources = [doc.metadata.get("source", None) for doc, _score in results]
		formatted_response = f"Odgovor:{resp}\nIzvori: {sources}"
		print(formatted_response)


if __name__ == "__main__":
	main()
