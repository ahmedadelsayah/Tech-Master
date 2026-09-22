import os
from dotenv import load_dotenv
import cohere
from cohere.core.api_error import ApiError
from src.utils import check_api_key


load_dotenv()
API_KEY = os.getenv("COHERE_API_KEY")

check_api_key(API_KEY)


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

    except ApiError as e:
        status = getattr(e, "status_code", None)

        if status == 401:
            return "Authentication failed. Please check your API key."
        elif status == 429:
            return "Rate limit reached. Please wait a moment and try again."
        elif status == 404:
            return "The requested model is unavailable. Please contact support."
        else:
            print(f"API Error ({status}): {e}")
            return "Sorry, something went wrong with the AI service. Please try again later."

    except Exception as e: 
        print(f"Error generating response: {e}")
        return "sorry, someting went wrong while generating the response. Please try again later."


if __name__ == "__main__":
    test_messages = [
        {"role": "user", "content": "Tell me a fun fact about Python in one sentence."}
    ]
    print(get_ai_response(test_messages))