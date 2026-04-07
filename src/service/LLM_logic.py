from datetime import datetime
import os
from unittest import result
from dotenv import load_dotenv
from google import genai
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.messages import AIMessage
from langchain.agents import create_agent
from src.rag.rag import askllm
from src.service.schedule_manage import DEFAULT_SLOT_MINUTES, book_doctor_appointment
if __name__ != "__main__":
    from src.service.AskLLM import generate_answer

# LLM_HOST = os.getenv("OLLAMA_HOST")
LLM_HOST = os.getenv("OLLAMA_HOST_API")
LLM_API_KEY = os.getenv("OLLAMA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI = genai.Client(api_key=GEMINI_API_KEY)

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
def doctor_appointment(doctor_name: str,
	patient_name: str,
	requested_start: datetime,
	duration_minutes: int = DEFAULT_SLOT_MINUTES,
) -> str:
    """
	Book an appointment if the slot is free; otherwise return alternatives.
    Args:
        doctor_name (str): The name of the doctor not include Dr. just the name.
        patient_name (str): The name of the patient.
        requested_start (datetime/str): The requested start time for the appointment Should be a datetime object like "dd-mm-yyyy Hour:Minute (00:00)".
        duration_minutes (int, optional): The duration of the appointment in minutes. Defaults to DEFAULT_SLOT_MINUTES.
    Returns:
        str: A message indicating whether the appointment was booked or if alternatives are provided.
	"""
    try:
        print(f"Tool received appointment request: doctor={doctor_name}, patient={patient_name}, start={requested_start}, duration={duration_minutes} minutes")
        result = book_doctor_appointment(doctor_name, patient_name, requested_start, duration_minutes)
        return result
    except Exception as e:        
        return f"Error: {str(e)}"

Agents_llm = ChatOllama(model="llama3.2:3b", baseurl=LLM_HOST, apikey=LLM_API_KEY).bind_tools([answer_medical_question, doctor_appointment])

def AgentsicAI(user_question: str) -> tuple[str, Exception]:
    response = GEMINI.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=f"Translate the following sentence from Thai to English Just translate and don't add anything else:\n{user_question}")
    translated_user_question = response.text
    print(f"Translated user question: {translated_user_question}")
    messages = [
        (
            "system",
            "You are a helpful assistant that answer the user questions. Use the tool to answer the question if needed.",
        ),
        ("human", translated_user_question),
    ]
    # print(f"User question: {user_question}")
    try :
        print("Invoking LLM with tools...")
        ai_msg = Agents_llm.invoke(messages)
        print(ai_msg.content)
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
                    tool_result, err = askllm(tool_call['args']['user_question'], user_question)
                    if err is not None:
                        return None, err
                    return tool_result, None
                elif tool_name == "doctor_appointment":
                    print("Calling doctor_appointment tool...")
                    tool_result = book_doctor_appointment(
                        doctor_name=tool_call['args']['doctor_name'],
                        patient_name=tool_call['args']['patient_name'],
                        requested_start=tool_call['args']['requested_start']
                    )
                    print(f"Tool Result: {tool_result}")
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
