from llms.llm_client import get_llm_client
from llms.model_registry import GeneratorModels
from config.settings import settings
import sys

def test_connection():
    try:
        settings.validate()
        print("Configuration valid.")
        
        model_name = GeneratorModels.MISTRAL_SMALL
        print(f"Connecting to OpenRouter with model {model_name}...")
        
        client = get_llm_client(model_name)
        
        response = client.generate(
            messages=[
                {"role": "user", "content": "Say 'Connection Successful' and nothing else."}
            ]
        )
        
        print(f"Response received: {response}")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_connection()
    if not success:
        sys.exit(1)
