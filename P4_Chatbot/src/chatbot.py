from typing import Dict, List

from src.prompts import (
    ROLE_USER,
    ROLE_ASSISTANT,
    build_messages,
    messages_to_text,
    is_valid_messages,
    get_fallback_message,
    clean_user_input,
    MAX_HISTORY_MESSAGES,
)

from src.api_client import get_ai_response

EXIT_COMMANDS = ("exit", "quit", "bye")

def get_user_input() -> str:
    """Read one message from the user."""

    try:
        return input("\nYou: ").strip()
    except (KeyboardInterrupt, EOFError):
        return "exit"


def display_response(response: str) -> None:
    """Display the AI response in a consistent format."""

    print(f"\nAssistant: {response}")


def is_exit_command(user_input: str) -> bool:
    """Return True if the user wants to end the conversation."""

    return user_input.lower().strip() in EXIT_COMMANDS


def run_chatbot() -> None:
    """Run the interactive chatbot conversation."""

    history: List[Dict[str, str]] = []

    print("=" * 55)
    print("        Welcome to TechMaster Assistant!")
    print("        Your Friendly Programming Tutor")
    print("=" * 55)
    print("Ask me programming or technology questions.")
    print("Type 'exit', 'quit', or 'bye' to end the chat.")

    while True:
        raw_input = get_user_input()

        if is_exit_command(raw_input):
            print("\nAssistant: Goodbye!")
            break

        try:
            user_input = clean_user_input(raw_input)

        except ValueError as error:
            print(f"\nAssistant: {error}")
            continue

        messages = build_messages(
            user_input=user_input,
            history=history,
        )

        if not is_valid_messages(messages):
            print(f"\nAssistant: {get_fallback_message()}")
            continue

        response = get_ai_response(messages)

        if not response or not response.strip():
            print(f"\nAssistant: {get_fallback_message()}")
            continue

        response = response.strip()

        display_response(response)

        history.append({"role": ROLE_USER,"content": user_input})

        history.append({"role": ROLE_ASSISTANT,"content": response})

        if len(history) > MAX_HISTORY_MESSAGES:
            history = history[-MAX_HISTORY_MESSAGES:]

if __name__ == "__main__":
    run_chatbot()