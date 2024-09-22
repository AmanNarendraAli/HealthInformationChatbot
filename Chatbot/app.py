from embeddings import check_for_embeddings, create_embeddings
from chain import build_chain

def main():
    # Check if embeddings exist
    if not check_for_embeddings():
        print("Embeddings not found. Creating embeddings...")
        create_embeddings()
    else:
        print("Embeddings found.")

    # Build the RAG chain
    rag_chain = build_chain()

    # Main interaction loop
    while True:
        question = input("Enter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        answer = rag_chain.invoke({"question": question})
        print("Answer:", answer)

if __name__ == "__main__":
    main()