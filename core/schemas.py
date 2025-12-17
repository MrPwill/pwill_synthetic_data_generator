from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

class GenerationRequest(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str
    data_type: str = Field(..., description="Type of data: tabular, json, text, etc.")
    num_samples: int = Field(default=1, ge=1, le=50)
    schema_def: Optional[Dict[str, Any]] = Field(default=None, description="JSON schema if applicable")
    model_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class GeneratedSample(BaseModel):
    content: Union[str, Dict[str, Any]]
    
class GenerationResult(BaseModel):
    request_id: str
    samples: List[GeneratedSample]
    raw_output: str
    model_used: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class EvaluationCriteria(BaseModel):
    correctness: bool = True
    schema_compliance: bool = True
    diversity: bool = False
    
class Feedback(BaseModel):
    score: int = Field(..., ge=0, le=100)
    comments: str
    passed: bool
