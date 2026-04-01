import os
import sys
from pathlib import Path

from langchain_ollama import ChatOllama
from google import genai
from dotenv import load_dotenv
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retrive_data import load_data, split_texts, split_data, OCR_load_data
from src.rag.embedding_data import embed_text, setup_chroma_db, query_chuncks


load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI = genai.Client(api_key=GEMINI_API_KEY)

llm = ChatOllama(model="medllama2:7b", baseurl=OLLAMA_HOST)

def askllm(query: str, user_messages: str)-> tuple[str, Exception]:
    print("Asking LLM...")
    collection = setup_chroma_db()
    # print(f"User message: {user_messages}")
    # PROMPT_CONTEXT = ""
    response = GEMINI.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=f"Translate the following sentence from English to Thai Just translate and don't add anything else:\n{query}")
    query = response.text
    print(f"Translated user question for retrieval: {query}")
    results = query_chuncks(query, collection)   
    PROMPT_CONTEXT = results['documents']
    translated_context = ""
    try: 
        response = GEMINI.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=f"Translate the following context from Thai to English just translate and don't add anything else:\n{PROMPT_CONTEXT}")
        # print(f"Translation response: {response.text}")
        translated_context += response.text + "\n"
    except Exception as e:
        print(f"Error during translation: {e}")
        return None, e
    print("Successfully translated context.")
    try:
        response = GEMINI.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=f"Translate the following sentence from Thai to English Just translate and don't add anything else:\n{user_messages}")
        Quesions = response.text
    except Exception as e:
        print(f"Error during user message translation: {e}")
        return None, e
    print(f"Translated user question: {Quesions}")
    # print(f"Retrieved context: {PROMPT_CONTEXT}")
    # messages = [
    #     (
    #         "system",
    #         f"""You are a helpful assistant that answer the user questions. Use the following context from the documents to provide accurate answers:\n
    #         This is the context you have retrieved from the documents:\n
    #         {PROMPT_CONTEXT} , if the question come with choice, please answer with the best one. if it doesn't come with choice, just answer the question based on the context and explain the context. if you don't know the answer, just say you don't know don't try to make up an answer.""",
    #     ),
    #     ("human", user_messages),
    # ]
    messages = f"""You are a helpful assistant that answer the user questions. Use the following context from the documents to provide accurate answers:\n
           This is the context you have retrieved from the documents:\n
            {translated_context} , 
            if the question come with choice, please answer with the best one. 
            if it doesn't come with choice, just answer the question based on the context and explain the context. 
            if you don't know the answer, just say you don't know don't try to make up an answer.
            when you answer the question say Our Recommendation is: before the answer.
            User question: {Quesions}"""
    print(f"Sending request with messages: {messages}")
    # print(type(messages))
    try : 
        # ai_msg = llm.invoke(messages)
        print("askllm")
        ai_msg = requests.post('https://swuai.swu.ac.th/swu/api/service/chat', headers={'Authorization': f'Bearer {os.getenv("SWU_AI_API_KEY")}'}, json={"user_id": os.getenv("SWU_AI_USER_ID"), "model": "openai/gpt-5", "content": messages})
        # print(f"LLM Response: {ai_msg.content}")
        ai_msg_content = ai_msg.json().get('choices')[0].get('message').get('content')
        translated_response = GEMINI.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=f"Translate the following sentence from English to Thai Just translate and don't add anything else:\n{ai_msg_content}")
        print(f"Translated LLM response: {translated_response.text}")
        return translated_response.text, None
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        return None, e


def setup_rag():
    # Load and split documents
    print("Setting up RAG...")
    collection = setup_chroma_db()
    exist_ids = collection.get()['ids']
    
    documents = OCR_load_data()
    # print(type(documents))

    # texts = split_data(documents)
    # print("Successfully split documents into chunks.")
    texts = split_texts(documents)
    # print("Text: ", texts[1:10])
    embedding = embed_text(texts)

    for i, text in enumerate(texts):
        if f"doc_{i}" in exist_ids:
            print(f"Document doc_{i} already exists in the database. Skipping insertion.")
            continue
        collection.upsert(
            ids=[f"doc_{i}"],
            documents=[text],
            embeddings=[embedding[i]]
        )
        print(f"Inserted document doc_{i} into the database.")
    # result = askllm("สิทธิหลักประกันสุขภาพแห่งชาติคืออะไร", collection)
    # print(f"LLM Response: {result}")
    return collection


if __name__ == "__main__":
    # print("Running RAG setup...")
    # collection = setup_rag()
    collection = setup_chroma_db()
    try:
        msg, err = askllm("วิธีการดูแลน้ำนมให้มีคุณภาพ","การดูแลน้ำนมให้มีคุณภาพควรทำอย่างไร")
        if err is not None:
            print(f"Error in askllm: {str(err)}")
        else:
            print(f"LLM Response: {msg}")
    except Exception as e:
        print(f"Error in askllm: {str(e)}")
    