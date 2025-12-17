import logging
from typing import List, Dict, Optional, Any, Union
from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenRouterClient:
    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: Optional[int] = None):
        self.client = OpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
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
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dicts (role, content)
            json_mode: Whether to force JSON output
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            The content of the response message.
        """
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
                # OpenRouter/OpenAI specific for JSON mode
                params["response_format"] = {"type": "json_object"}

            logger.info(f"Generating with model {self.model_name}...")
            response = self.client.chat.completions.create(**params)
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Received empty response from LLM")
                
            return content

        except Exception as e:
            logger.error(f"Error executing LLM call: {e}")
            raise
