import chromadb
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from retriever import CustomChromaRetriever

def build_chain():

    collection = chromadb.PersistentClient(path="./chroma_db").get_collection("nfcorpus")

    retriever = CustomChromaRetriever(collection=collection, k_results=3)

    prompt_template =  """You are a medical information retrieval system. Given a health-related query,
    Use the following documents to answer the question.
    If you don't know the answer, just say that you don't know.
    Use five sentences maximum and keep the answer concise:
    Question: {question}
    Documents: {documents}
    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question", "documents"],
    )

    llm = ChatOllama(
        model="llama3.1",
        temperature=0,
    )

    rag_chain = prompt | retriever | llm | StrOutputParser()

    response = {"documents": retriever, "question": RunnablePassthrough()} | rag_chain
    return response

if __name__ == "__main__":
    chain = build_chain()
    question = "What is the treatment for hypertension?"
    answer = chain.invoke({"question": question})
    print(answer)
    
