from typing import List
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection

class CustomChromaRetriever(BaseRetriever):
    model:SentenceTransformer
    collection:Collection
    k_results:int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Encode the query
        query_embedding = self.model.encode(query, prompt_name="s2p_query").tolist()

        # Query the collection
        results = self.retrieve_from_collection(query_embedding)
        documents = []
        for doc, metadata, distance in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
           documents.append(Document(
               page_content=doc,
               metadata={**metadata, "score": 1 - distance}  # Convert distance to similarity score
           ))

        return documents
    
    def retrieve_from_collection(self, query_embedding):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.k_results,
            include=['documents', 'metadatas', 'distances']
        )

def get_retriever(collection_name: str = "nfcorpus", model_name: str = "dunzhang/stella_en_1.5B_v5", k: int = 3):
    return CustomChromaRetriever(collection_name, model_name, k)

