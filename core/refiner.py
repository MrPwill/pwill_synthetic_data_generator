import logging
import copy
from typing import List, Optional
from core.schemas import GenerationRequest, GenerationResult, EvaluationCriteria
from core.generator import GeneratorAgent
from core.judge import JudgeAgent
from llms.model_registry import JudgeModels

logger = logging.getLogger(__name__)

class RefinerAgent:
    def __init__(self, generator: GeneratorAgent, judge_model: str = JudgeModels.LLAMA_3_1_405B):
        self.generator = generator
        self.judge = JudgeAgent(model_name=judge_model)
        
    def generate_verified(self, 
                          request: GenerationRequest, 
                          max_retries: int = 3,
                          criteria: Optional[EvaluationCriteria] = None) -> GenerationResult:
        """
        Generate data, evaluate it, and regenerate if necessary.
        """
        if criteria is None:
            criteria = EvaluationCriteria()
            
        current_request = request
        attempt = 0
        
        while attempt <= max_retries:
            attempt += 1
            logger.info(f"Generation attempt {attempt}/{max_retries + 1}")
            
            # 1. Generate
            result = self.generator.generate(current_request)
            
            # 2. Evaluate
            feedbacks = self.judge.evaluate(
                samples=result.samples,
                original_prompt=request.prompt,  # Always judge against original intent
                criteria=criteria,
                schema=request.schema_def
            )
            
            # Check if all passed
            all_passed = all(f.passed for f in feedbacks)
            
            if all_passed:
                logger.info("All samples passed evaluation.")
                return result
            
            if attempt > max_retries:
                logger.warning("Max retries reached. Returning last result.")
                return result
                
            # 3. Refine Prompt (Simple strategy: Append feedback to prompt)
            # We construct a new prompt for the next iteration
            logger.info("Refining prompt based on feedback...")
            
            failed_feedback_summary = "\n".join(
                [f"- {f.comments}" for f in feedbacks if not f.passed]
            )
            
            refined_prompt = f"{request.prompt}\n\nPREVIOUS ATTEMPT FAILED. FEEDBACK:\n{failed_feedback_summary}\n\nFIX THE ISSUES."
            
            # Update request for next loop
            # We use copy to not mutate original if intended to reuse
            current_request = copy.deepcopy(request)
            current_request.prompt = refined_prompt
            
        return result
