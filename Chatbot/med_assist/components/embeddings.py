import chromadb
from beir.datasets.data_loader import GenericDataLoader
from beir.util import download_and_unzip
from sentence_transformers import SentenceTransformer
from med_assist.config import CONFIG

def format_document(doc_data: dict):

    title = doc_data.get('title')
    text = doc_data.get('text')

    return f"{title}; {text}"

def check_for_embeddings():
    """Checks if embeddings exist"""
    
    collections = chromadb.PersistentClient(path = CONFIG['chromadb']['path']).list_collections()

    collection_names = [collection.name for collection in collections]
    
    embeddings_exist = CONFIG['chromadb']['collection'] in collection_names
    
    return embeddings_exist

def create_embeddings():
    """Downloads dataset, create embeddings and loads them into vector database."""
    # Download dataset

    data_path = download_and_unzip(
        url = CONFIG['beir']['url'],
        out_dir = CONFIG['beir']['path']
        )
    corpus, _, _ = GenericDataLoader(
        data_folder = data_path
        ).load(split="train")

    # Prepare keys and documents for a vector database

    ids = list(corpus.keys())
    docs = [format_document(corpus.get(id)) for id in ids]

    # Prepare embedding model
    emb_model = SentenceTransformer("dunzhang/stella_en_1.5B_v5")

    # Embed documents

    doc_embeddings = emb_model.encode(docs, show_progress_bar=True).tolist()

    # Load embeddings to a vector database 
    client = chromadb.PersistentClient(path=CONFIG['chromadb']['path'])
    collection = client.get_or_create_collection(name=CONFIG['chromadb']['collection'])
    
    
    collection.add(
        ids=ids,
        embeddings=doc_embeddings,
        documents=docs
    )


if __name__ == "__main__":
    create_embeddings()