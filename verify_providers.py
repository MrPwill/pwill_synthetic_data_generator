import asyncio
import logging
from llms.llm_client import get_llm_client
from llms.model_registry import GeneratorModels, JudgeModels, MODEL_FRIENDLY_NAMES

# Configure logging to see output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_provider(model_id, model_name):
    print(f"\n--- Testing {model_name} ({model_id}) ---")
    try:
        client = get_llm_client(model_id)
        # Simple hello world
        response = client.generate(
            messages=[{"role": "user", "content": "Say 'Health Check Passed' and nothing else."}],
            max_tokens=20
        )
        print(f"✅ SUCCESS: {response}")
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")

async def main():
    # Test one model from each provider
    
    # 1. Llama
    await test_provider(GeneratorModels.LLAMA_3_3_70B, "Llama 3.3 70B")
    
    # 2. DeepSeek
    await test_provider(GeneratorModels.DEEPSEEK_R1, "DeepSeek R1")
    
    # 3. Qwen
    await test_provider(GeneratorModels.QWEN_3_CODER, "Qwen 3 Coder")
    
    # 4. Gemma
    await test_provider(GeneratorModels.GEMMA_3_27B, "Gemma 3 27B")

if __name__ == "__main__":
    asyncio.run(main())
