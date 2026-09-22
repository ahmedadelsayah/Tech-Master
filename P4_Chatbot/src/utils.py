
import sys


def check_api_key(api_key: str) -> None:
    """Stop the program early with a clear message if the API key is missing."""
    if not api_key:
        print("Missing API key. Please set COHERE_API_KEY in your .env file.")
        sys.exit(1)