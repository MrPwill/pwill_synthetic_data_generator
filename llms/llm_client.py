import logging
from typing import List, Dict, Optional, Any, Union
import google.generativeai as genai
from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config.settings import settings
from llms.model_registry import LLMProvider, get_model_provider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedLLMClient:
    """Interface for LLM clients."""
    def generate(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        raise NotImplementedError

class OpenAICompatibleClient(UnifiedLLMClient):
    def __init__(self, model_name: str, base_url: str, api_key: str, temperature: float = 0.7, max_tokens: Optional[int] = None):
        if not api_key:
            logger.warning(f"API key missing for model {model_name} (Base URL: {base_url})")
            
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @retry(
        retry=retry_if_exception_type((APITimeoutError, RateLimitError, APIError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        try:
            params: Dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.temperature,
            }
            
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            elif self.max_tokens is not None:
                params["max_tokens"] = self.max_tokens

            if json_mode:
                params["response_format"] = {"type": "json_object"}

            logger.info(f"Generating with model {self.model_name} via OpenAI compatible client...")
            response = self.client.chat.completions.create(**params)
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Received empty response from LLM")
                
            return content

        except Exception as e:
            logger.error(f"Error executing LLM call: {e}")
            raise

class GoogleClient(UnifiedLLMClient):
    def __init__(self, model_name: str, api_key: str, temperature: float = 0.7, max_tokens: Optional[int] = None):
        if not api_key:
            logger.warning(f"API key missing for Google model {model_name}")
        
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Determine generation config
        self.generation_config = genai.types.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens
        )
        self.model = genai.GenerativeModel(model_name=model_name)

    @retry(
        retry=retry_if_exception_type(Exception), # Broad retry for now, narrow down later
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        try:
            # Convert messages to Google format
            # Google Generative AI supports a list of content dicts or chat history.
            # Simple conversion: 
            # System prompt -> configuration or separate handling (Google system instructions)
            # User/Assistant -> history
            
            system_instruction = None
            history = []
            last_user_message = ""
            
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    system_instruction = content
                elif role == "user":
                    last_user_message = content # We'll send the last one as the triggers
                elif role == "assistant":
                    # For history, Google expects 'user' or 'model' roles
                    history.append({"role": "model", "parts": [content]})
                    
                # Note: This is a simplified chat conversion. 
                # Ideally we build a chat session if there's history, but for single generate call:
            
            # Re-initializing model with system instruction if present
            if system_instruction:
                 self.model = genai.GenerativeModel(model_name=self.model_name, system_instruction=system_instruction)
            
            # Override config if needed
            current_config = self.generation_config
            if temperature is not None or max_tokens is not None:
                current_config = genai.types.GenerationConfig(
                    temperature=temperature if temperature is not None else self.temperature,
                    max_output_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                    response_mime_type="application/json" if json_mode else "text/plain"
                )
            elif json_mode:
                 current_config = genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    response_mime_type="application/json"
                )

            logger.info(f"Generating with model {self.model_name} via Google client...")
            response = self.model.generate_content(last_user_message, generation_config=current_config)
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error executing Google LLM call: {e}")
            raise


class LLMClientFactory:
    @staticmethod
    def create(model_name: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> UnifiedLLMClient:
        provider = get_model_provider(model_name)
        if provider == LLMProvider.OPENROUTER:
            return OpenAICompatibleClient(
                model_name=model_name,
                base_url=settings.OPENROUTER_BASE_URL,
                api_key=settings.OPENROUTER_API_KEY,
                temperature=temperature,
                max_tokens=max_tokens
            )
        elif provider == LLMProvider.DEEPSEEK:
            return OpenAICompatibleClient(
                model_name=model_name,
                base_url=settings.DEEPSEEK_BASE_URL,
                api_key=settings.DEEPSEEK_API_KEY,
                temperature=temperature,
                max_tokens=max_tokens
            )
        elif provider == LLMProvider.NVIDIA:
            return OpenAICompatibleClient(
                model_name=model_name,
                base_url=settings.NVIDIA_BASE_URL,
                api_key=settings.NVIDIA_API_KEY,
                temperature=temperature,
                max_tokens=max_tokens
            )
        elif provider == LLMProvider.DASHSCOPE:
             return OpenAICompatibleClient(
                model_name=model_name,
                base_url=settings.DASHSCOPE_BASE_URL,
                api_key=settings.DASHSCOPE_API_KEY,
                temperature=temperature,
                max_tokens=max_tokens
            )
        elif provider == LLMProvider.GOOGLE:
            return GoogleClient(
                model_name=model_name,
                api_key=settings.GOOGLE_API_KEY,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            # Default fallback or error?
            logger.warning(f"Unknown provider for model {model_name}, defaulting to OpenRouter/OpenAI fallback if configured")
            # If we want to strictly follow request to STOP using openrouter, we might error here.
            # But let's fallback to OpenRouter just in case old models are passed.
            if settings.OPENROUTER_API_KEY:
                 return OpenAICompatibleClient(
                    model_name=model_name,
                    base_url=settings.OPENROUTER_BASE_URL,
                    api_key=settings.OPENROUTER_API_KEY,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            raise ValueError(f"No provider configured for model {model_name}")

def get_llm_client(model_name: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> UnifiedLLMClient:
    return LLMClientFactory.create(model_name, temperature, max_tokens)
