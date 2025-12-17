from sqlalchemy.orm import Session
from memory.database import DBGeneration
from core.schemas import GenerationRequest, GenerationResult
import json

class GenerationRepository:
    def __init__(self, db: Session):
        self.db = db
        
    def save_result(self, request: GenerationRequest, result: GenerationResult, avg_score: float = 0.0):
        # Convert samples to serializable format
        samples_data = [
            s.dict() if hasattr(s, 'dict') else s.__dict__ 
            for s in result.samples
        ]
        
        db_item = DBGeneration(
            id=result.request_id,
            prompt=request.prompt,
            data_type=request.data_type,
            model_name=result.model_used,
            samples=samples_data,
            average_score=avg_score,
            created_at=result.timestamp
        )
        self.db.add(db_item)
        self.db.commit()
        self.db.refresh(db_item)
        return db_item
        
    def get_history(self, limit: int = 50):
        return self.db.query(DBGeneration).order_by(DBGeneration.created_at.desc()).limit(limit).all()
