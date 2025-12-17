import logging
import json
from typing import List, Optional
from core.schemas import GeneratedSample, Feedback, EvaluationCriteria
from llms.openrouter_client import OpenRouterClient
from evaluation.metrics import validate_json_schema

logger = logging.getLogger(__name__)

class JudgeAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def evaluate(self, 
                 samples: List[GeneratedSample], 
                 original_prompt: str, 
                 criteria: EvaluationCriteria,
                 schema: Optional[dict] = None) -> List[Feedback]:
        """
        Evaluate a list of samples against criteria.
        """
        feedbacks = []
        client = OpenRouterClient(model_name=self.model_name)
        
        for sample in samples:
            # 1. Hard check: Schema validation
            if schema and isinstance(sample.content, (dict, list)):
                if not validate_json_schema(sample.content, schema):
                    feedbacks.append(Feedback(
                        score=0, 
                        comments="Failed JSON Schema Validation", 
                        passed=False
                    ))
                    continue

            # 2. LLM Evaluation
            # We treat content as string for the prompt
            content_str = json.dumps(sample.content) if isinstance(sample.content, (dict, list)) else str(sample.content)
            
            prompt = self._build_judge_prompt(content_str, original_prompt, criteria)
            
            try:
                response = client.generate(
                    messages=[{"role": "user", "content": prompt}],
                    json_mode=True
                )
                
                eval_data = json.loads(response)
                feedbacks.append(Feedback(
                    score=eval_data.get("score", 0),
                    comments=eval_data.get("feedback", "No feedback provided"),
                    passed=eval_data.get("score", 0) >= 70 # Threshold
                ))
            except Exception as e:
                logger.error(f"Judge evaluation failed: {e}")
                feedbacks.append(Feedback(score=0, comments=f"Error: {e}", passed=False))
                
        return feedbacks

    def _build_judge_prompt(self, content: str, original_prompt: str, criteria: EvaluationCriteria) -> str:
        return f"""
        You are an impartial judge evaluating synthetic data.
        
        Original User Request: {original_prompt}
        
        Generated Content to Evaluate:
        {content}
        
        Evaluation Criteria:
        - Correctness: {criteria.correctness}
        - Schema Compliance: {criteria.schema_compliance}
        - Diversity/Creativity: {criteria.diversity}
        
        Provide your evaluation in JSON format:
        {{
            "score": <0-100 integer>,
            "feedback": "<detailed critique>"
        }}
        """
