if __name__ != "__main__":
    from src.rag.retrive_data import load_data, split_data
    from src.rag.embedding_data import embed_text, setup_chroma_db, query_chuncks
from langchain_ollama import ChatOllama
import os

OLLAMA_HOST = os.getenv("OLLAMA_HOST")

llm = ChatOllama(model="medllama2:7b", baseurl=OLLAMA_HOST)

def askllm(user_messages: str, collection):
    print(f"User message: {user_messages}")
    # PROMPT_CONTEXT = ""
    results = query_chuncks(user_messages, collection)   
    PROMPT_CONTEXT = results['documents']
    messages = [
        (
            "system",
            f"""You are a helpful assistant that answer the user questions. Use the following context from the documents to provide accurate answers:\n
            This is the context you have retrieved from the documents:\n
            {PROMPT_CONTEXT} """,
        ),
        ("human", user_messages),
    ]
    try : 
        ai_msg = llm.invoke(messages)
        return ai_msg.content, None
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        return None, e


def setup_rag():
    # Load and split documents
    print("Setting up RAG...")
    collection = setup_chroma_db()
    exist_ids = collection.get()['ids']
    
    documents = load_data()

    texts = split_data(documents)

    embedding = embed_text(texts)

    for i, text in enumerate(texts):
        if f"doc_{i}" in exist_ids:
            print(f"Document doc_{i} already exists in the database. Skipping insertion.")
            continue
        collection.upsert(
            ids=[f"doc_{i}"],
            documents=[text.page_content],
            embeddings=[embedding[i]]
        )
        print(f"Inserted document doc_{i} into the database.")
    # result = askllm("สิทธิหลักประกันสุขภาพแห่งชาติคืออะไร", collection)
    # print(f"LLM Response: {result}")
    return collection
if __name__ == "__main__":
    from retrive_data import load_data, split_data
    from embedding_data import embed_text, setup_chroma_db, query_chuncks
    print("Running RAG setup...")
    collection = setup_rag()