import os
from dotenv import load_dotenv
import cohere

load_dotenv()
API_KEY = os.getenv("COHERE_API_KEY")

client = cohere.ClientV2(API_KEY)

def get_ai_response(messages: list) -> str:
    try:
        response = client.chat(
            model='command-a-03-2025',
            messages=messages
        )
        return response.message.content[0].text.strip()
    except (AttributeError, IndexError) as e:
        return "sorry, I couldn't generate a response at this time. Please try again later."
    except Exception as e: 
        print(f"Error generating response: {e}")
        return "sorry, someting went wrong while generating the response. Please try again later."


if __name__ == "__main__":
    test_messages = [
        {"role": "user", "content": "Tell me a fun fact about Python in one sentence."}
    ]
    print(get_ai_response(test_messages))