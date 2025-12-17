from memory.database import init_db, SessionLocal
from memory.repository import GenerationRepository
from core.generator import GeneratorAgent
from core.refiner import RefinerAgent

class AppState:
    def __init__(self):
        # Initialize DB
        init_db()
        self.db = SessionLocal()
        self.repo = GenerationRepository(self.db)
        
        # Agents
        self.generator = GeneratorAgent()
        self.refiner = RefinerAgent(self.generator)
        
    def close(self):
        self.db.close()

# Singleton instance
app_state = AppState()
