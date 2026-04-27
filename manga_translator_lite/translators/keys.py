import os

from dotenv import load_dotenv

load_dotenv()

# OpenAI-compatible providers (chatgpt / openrouter / deepseek / groq / custom)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
OPENAI_HTTP_PROXY = os.getenv('OPENAI_HTTP_PROXY')

# Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash-002')
