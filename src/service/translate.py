import openai
import os

from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(
    api_key=os.getenv("TYPHOON_API_KEY"),
    base_url="https://api.opentyphoon.ai/v1"
)
MODEL = "typhoon-v2.5-30b-a3b-instruct"

def translate_to_english(text: str) -> tuple[str, Exception]:
    try: 
        response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates Thai to English."},
            {"role": "user", "content": f"Translate the following: {text}"}
        ],
        temperature=0.7,
        max_tokens=256
        )
        return response.choices[0].message.content, None
    except Exception as e:
        print(f"Error during translation: {e}")
        return None, e
    
    
def translate_to_thai(text: str) -> tuple[str, Exception]:
    try: 
        response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates English to Thai."},
            {"role": "user", "content": f"Translate the following: {text}"}
        ],
        temperature=0.7,
        max_tokens=256
        )
        return response.choices[0].message.content, None
    except Exception as e:
        print(f"Error during translation: {e}")
        return None, e