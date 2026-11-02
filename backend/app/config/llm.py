import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def get_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """
    Return a configured Gemini chat model.
    Model: gemini-2.5-flash (fast + cheap, good for routing/classification).
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in environment.")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=temperature,
        convert_system_message_to_human=True,
    )