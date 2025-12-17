import json
import logging
from typing import List, Dict, Any, Optional
from core.schemas import GenerationRequest, GenerationResult, GeneratedSample
from llms.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)

class GeneratorAgent:
    def __init__(self):
        pass
        
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Main entry point for generating data.
        """
        client = OpenRouterClient(model_name=request.model_name)
        
        system_prompt = self._build_system_prompt(request)
        user_prompt = self._build_user_prompt(request)
        
        is_json = request.data_type.lower() in ["json", "tabular"]
        
        try:
            raw_response = client.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                json_mode=is_json
            )
            
            samples = self._parse_response(raw_response, request.data_type, request.num_samples)
            
            return GenerationResult(
                request_id=request.id,
                samples=samples,
                raw_output=raw_response,
                model_used=request.model_name
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def _build_system_prompt(self, request: GenerationRequest) -> str:
        base = "You are a highly advanced synthetic data generator. Your goal is to produce high-quality, diverse, and realistic data."
        
        if request.data_type == "json":
            base += "\nYou must output VALID JSON only. Do not include markdown fencing like ```json."
            if request.schema_def:
                base += f"\nFollow this JSON schema strictly:\n{json.dumps(request.schema_def, indent=2)}"
        elif request.data_type == "tabular":
            base += "\nOutput data as a list of JSON objects, which will be converted to CSV/Table. VALID JSON list only."
        else:
            base += "\nFollow the user's instructions precisely."
            
        return base

    def _build_user_prompt(self, request: GenerationRequest) -> str:
        prompt = f"Request: {request.prompt}\n"
        prompt += f"Number of samples to generate: {request.num_samples}\n"
        if request.data_type == "text":
            prompt += "Separate samples with '---' if multiple are requested."
        return prompt

    def _parse_response(self, raw_response: str, data_type: str, expected_count: int) -> List[GeneratedSample]:
        samples = []
        
        if data_type.lower() in ["json", "tabular"]:
            try:
                # cleaner parsing if markdown fences exist
                cleaned = raw_response.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                
                parsed = json.loads(cleaned)
                
                if isinstance(parsed, list):
                    for item in parsed:
                        samples.append(GeneratedSample(content=item))
                elif isinstance(parsed, dict):
                    # Maybe wrapped in a key? or single item
                    samples.append(GeneratedSample(content=parsed))
                else:
                    samples.append(GeneratedSample(content=str(parsed)))
                    
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response. Returning raw string.")
                samples.append(GeneratedSample(content=raw_response))
        else:
            # Text split
            if "---" in raw_response:
                parts = raw_response.split("---")
                for p in parts:
                    if p.strip():
                        samples.append(GeneratedSample(content=p.strip()))
            else:
                 samples.append(GeneratedSample(content=raw_response))
                 
        return samples
