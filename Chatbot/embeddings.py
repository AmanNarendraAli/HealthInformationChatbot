from langchain_community.vectorstores import FAISS
from beir.datasets.data_loader import GenericDataLoader
from beir.util import download_and_unzip
from sentence_transformers import SentenceTransformer
import chromadb

def format_document(doc_data: dict):

    title = doc_data.get('title')
    text = doc_data.get('text')

    return f"{title}; {text}"


query_prompt_name = "s2p_query"


def check_for_embeddings():
     # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        collection = client.get_collection("nfcorpus")
        return collection.count() > 0
    except ValueError:
        return False 

def create_embeddings():
    
    """Downloads dataset, create embeddings and loads them into vector database."""
    # Download dataset

    data_path = download_and_unzip(
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
    out_dir = "data/beir_datasets/"
    )
    data_folder = "data/beir_datasets/nfcorpus"  # Update this path
    loader = GenericDataLoader(data_folder=data_folder)
    corpus, queries, _= loader.load(split="train")


# Prepare keys and documents for a vector database

    docs_ids = list(corpus.keys())
    docs = [format_document(corpus.get(id)) for id in docs_ids]

# Prepare embedding model

    model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True).cuda()
    query_embeddings = model.encode(queries, prompt_name=query_prompt_name)
    doc_embeddings = model.encode(docs, show_progress_bar=True).tolist()
    print(query_embeddings.shape, doc_embeddings.shape)

    similarities = model.similarity(query_embeddings, doc_embeddings)
    print(similarities)

    # Load embeddings into vector database
    client = chromadb.PersistentClient(path="./chroma_db") 
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name="nfcorpus"
    )

    # Add documents to collection
    collection.add(
        embeddings=doc_embeddings,
        documents=docs,
        metadatas=[{"source": id} for id in docs_ids],
        ids=docs_ids
    )

if __name__ == "__main__":
    create_embeddings()