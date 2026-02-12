from langchain_ollama import ChatOllama
from langchain.tools import tool
import os
from src.rag.embedding_data import setup_chroma_db
from src.rag.rag import query_chuncks


collection = setup_chroma_db()

def generate_answer(query: str, user_messages: str = "") -> tuple[str, Exception]:
    print(f"User message: {query}")
    results = query_chuncks(query, collection)  
    PROMPT_CONTEXT = results['documents']
    OLLAMA_HOST = os.getenv("OLLAMA_HOST")
    llm = ChatOllama(model="llama3.2:3b", baseurl=OLLAMA_HOST)
    print(f"Original question: {user_messages}")
    messages = [
        (
            "system",
            f"""You are a helpful assistant that answer the user questions and translate them to Thai. Use the following context from the documents to provide accurate answers:\n
            This is the context you have retrieved from the documents:\n
            {PROMPT_CONTEXT} """,
        ),
        ("human", user_messages),
    ]
    try : 
        initial_answer = llm.invoke(messages)
        print(f"Initial answer: {initial_answer.content}")
        # return initial_answer.content, None
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        return None, e
    feedback_messages = [
        ("system",f"""You are the assistant that expert at meta-reasoning and self-evaluation of answers. 
         If the answer is correct, just return the same answer without any changes.
         Your task is to review the following answer based on the provided context and user question. Identify any potential mistakes, inaccuracies, or areas 
         for improvement in the answer. Provide constructive feedback to help enhance the quality of the response. and made answer more complete 
         if needed. 
         The context you have is:\n {PROMPT_CONTEXT} \n The user question is:\n {user_messages}"""),
        ("human", f"""{initial_answer.content}"""),
    ]
    try:
        feedback = llm.invoke(feedback_messages)
        print(f"Feedback: {feedback.content}")
    except Exception as e:
        print(f"Error during feedback invocation: {e}")
        return None, e
    EXPAND_PROMPT = f"""You are a helpful assistant that answer the user questions. Use the following context from the documents to provide accurate answers:\n
    {PROMPT_CONTEXT} \n and also use the following feedback to improve your answer:\n {feedback.content}"""
    final_messages = [
        ("system", EXPAND_PROMPT),
        ("human", user_messages),
    ]
    try: 
        final_answer = llm.invoke(final_messages)
        print(f"Final answer before translation: {final_answer.content}")
        Thai_answer = llm.invoke([
            ("system", "You are a helpful assistant that translate the text to Thai language accurately and fluently."),
            ("human", final_answer.content),
        ])
        print(f"Final Thai answer: {Thai_answer.content}")
        return Thai_answer.content, None
    except Exception as e:
        print(f"Error during final LLM invocation: {e}")
        return None, e