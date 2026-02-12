import os
from unittest import result
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.messages import AIMessage
from langchain.agents import create_agent
from src.rag.rag import askllm
if __name__ != "__main__":
    from src.service.AskLLM import generate_answer

LLM_HOST = os.getenv("OLLAMA_HOST")

@tool
def answer_medical_question(user_question: str) -> str:
    """
    Tool to answer medical questions using the LLM. pharse the question and return the answer.
    Args:
        user_question (str): User input.
    Returns:
        str: The answer to the medical question.
        error: If any error occurs during the process.
    """
    try :
        print(f"Tool received question: {user_question}")
        answer, err = generate_answer(user_question)
        if err is not None:
            return f"Error: {str(err)}"
        return answer
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def doctor_appointment(date: str, departments: str) -> str:
    '''Schedule a doctor appointment on the given date.'''
    return f"Doctor appointment scheduled for {date} in department {departments}."

Agents_llm = ChatOllama(model="llama3.2:3b", baseurl=LLM_HOST).bind_tools([answer_medical_question, doctor_appointment])

def AgentsicAI(user_question: str) -> tuple[str, Exception]:
    messages = [
        (
            "system",
            "You are a helpful assistant that answer the user questions. Use the tool to answer the question if needed.",
        ),
        ("human", user_question),
    ]
    try :
        ai_msg = Agents_llm.invoke(messages)
        # print(ai_msg.content)
        if isinstance(ai_msg, AIMessage) and ai_msg.tool_calls:
            # print(ai_msg.tool_calls)
            if ai_msg.tool_calls:
                # If the LLM made a tool call, execute the tool and get the result
                tool_call = ai_msg.tool_calls[0]
                print(f"Tool Call: {tool_call}")
                print(f"Tool Name: {tool_call['name']}")
                tool_name = tool_call['name']
                if tool_name == "answer_medical_question":
                    print("Calling answer_medical_question tool...")
                    tool_result, err = generate_answer(tool_call['args']['user_question'], user_question)
                    if err is not None:
                        return None, err
                    return tool_result, None
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        return None, e

if __name__ == "__main__":
    from AskLLM import generate_answer
    question = input("Enter your medical question: ")
    answer, err = AgentsicAI(question)
    if err is not None:
        print(f"Error: {str(err)}")
    else:
        print(f"Answer: {answer}")
