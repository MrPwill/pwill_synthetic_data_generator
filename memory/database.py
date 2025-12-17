from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from config.settings import settings

Base = declarative_base()

class DBGeneration(Base):
    __tablename__ = "generations"
    
    id = Column(String, primary_key=True)
    prompt = Column(Text, nullable=False)
    data_type = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    
    # Store raw result or list of samples as JSON
    samples = Column(JSON, nullable=True) 
    
    # Validation info
    average_score = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=datetime.utcnow)

engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
