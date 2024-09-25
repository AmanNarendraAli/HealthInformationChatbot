from fastapi import FastAPI
from langserve import add_routes
from med_assist.chain import build_chain
from dotenv import load_dotenv
from med_assist.components.embeddings import check_for_embeddings, create_embeddings

load_dotenv()

app = FastAPI(
    title="med_assist",
    version="0.2.0",
    description="Medical assistant API"
)

if not check_for_embeddings():
    create_embeddings()

chain = build_chain()

add_routes(
    app=app,
    runnable=chain,
    path="/med_assist"
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)