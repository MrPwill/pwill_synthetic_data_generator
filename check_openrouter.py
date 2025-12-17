from openai import OpenAI
from config.settings import settings
import sys

def test_connection():
    try:
        settings.validate()
        print("Configuration valid.")
        
        print(f"Connecting to OpenRouter...")
        client = OpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
        )
        
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free", # Using a potentially free/cheap model for test
            messages=[
                {"role": "user", "content": "Say 'Connection Successful' and nothing else."}
            ],
        )
        
        print(f"Response received: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    if not success:
        sys.exit(1)
