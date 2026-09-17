# TechMaster Assistant

A beginner-friendly programming and technology chatbot built with Python and the Cohere API.

## Project Overview

TechMaster Assistant is an interactive command-line chatbot designed to answer programming and technology questions clearly and simply.

The project uses:
- Python
- Cohere API
- `python-dotenv`
- Structured prompts
- Conversation history
- Input validation and fallback messages

## Project Structure

```text
project/
│
├── src/
│   ├── api_client.py
│   ├── chatbot.py
│   ├── prompts.py
│   └── utils.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Files Description

### `api_client.py`
Handles communication with the Cohere API.

It:
- Loads the API key from `.env`
- Creates the Cohere client
- Sends messages to the model
- Returns the generated AI response
- Handles API errors

The current client uses the Cohere model `command-a-03-2025`. 

### `chatbot.py`
Contains the main chatbot application.

It:
- Reads user input
- Checks exit commands
- Cleans and validates input
- Builds the conversation messages
- Sends the request to the AI
- Displays the response
- Stores recent conversation history

Supported exit commands:

```text
exit
quit
bye
```

### `prompts.py`
Contains the prompt system and conversation helpers.

It includes:
- System prompt
- Prompt templates
- Custom prompt builder
- Message builder
- History trimming
- Message validation
- User input cleaning
- Fallback message

The prompt system is designed for a beginner-friendly programming tutor.

### `utils.py`
Contains small reusable helper functions that can be shared by different parts of the project.

The goal is to keep common utility operations separate from the chatbot and API logic.

## Prompt Templates

The project currently supports these templates:

| Template | Purpose |
|---|---|
| `default` | Answer a normal question |
| `explain` | Explain a topic for a beginner |
| `summarize` | Summarize text into 3 bullet points |
| `explain_code` | Explain code step by step |

## Conversation History

The API does not keep conversation memory automatically, so recent messages are sent again with each request.

The project keeps a maximum of:

```text
20 messages
```

A single user message can contain up to:

```text
2000 characters
```

## Environment Setup

### 1. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install cohere python-dotenv
```

### 3. Create `.env`

Add your Cohere API key:

```env
COHERE_API_KEY=your_api_key_here
```

Do not upload or commit the `.env` file to GitHub.

## Running the Chatbot

From the project directory:

```bash
python -m src.chatbot
```

The chatbot will display:

```text
Welcome to TechMaster Assistant!
Your Friendly Programming Tutor
```

Then you can enter programming or technology questions.

## Example

```text
You: What is a loop?

Assistant: A loop is a programming structure that repeats a block
of code multiple times...
```

## Error Handling

The project handles:
- Empty user input
- Messages that are too long
- Invalid message structures
- Keyboard interruption
- API response errors
- Empty AI responses

When a response cannot be generated, a fallback message is displayed.

## Security

- Store the Cohere API key in `.env`
- Never hard-code API keys in Python files
- Add `.env` to `.gitignore`
- Do not share API keys publicly

## Technologies

- Python
- Cohere API
- python-dotenv
- Object-free modular Python design
- Command-line interface

## Future Improvements

Possible future improvements:
- Streamlit web interface
- Better conversation memory
- More prompt templates
- Chat export
- Response streaming
- Logging
- Unit tests
