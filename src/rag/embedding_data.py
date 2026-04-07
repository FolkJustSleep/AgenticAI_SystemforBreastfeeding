import chromadb
from langchain_ollama import OllamaEmbeddings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

CHROMA_PATH = r"data/chroma_db"
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:8b"
)

def setup_chroma_db():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection_name ="breast_feeding"
    # collection_name ="New_Medical_Articles"
    collection = chroma_client.get_or_create_collection(name=collection_name , embedding_function=OllamaEmbeddingFunction(model_name="qwen3-embedding",url="http://localhost:11434"))
    return collection

def embed_text(text: list[str]):
    # input_text = []
    # for i, content in enumerate(text):
    #     input_text.append(content.page_content)
        # print(f"Preparing text for embedding.[{i}]")
    response = embeddings.embed_documents(text)
    print("Embedding completed.")
    # print(f"Embedding response: {response}") 
    return response

def query_chuncks(query: str, collection):
    print(f"Querying chunks for: {query}")
    query_embedding = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )
    return results
