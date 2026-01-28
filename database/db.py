from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Use environment variable or default
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/quality_control")

Base = declarative_base()

class Defect(Base):
    __tablename__ = 'defects'
    
    id = Column(Integer, primary_key=True)
    defect_type = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_defect(defect_type: str, confidence: float):
    db = SessionLocal()
    try:
        defect = Defect(defect_type=defect_type, confidence=confidence)
        db.add(defect)
        db.commit()
        db.refresh(defect)
        return defect
    except Exception as e:
        print(f"Error saving defect: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    # Create tables if not exist (alternative to init.sql script usage via python)
    Base.metadata.create_all(bind=engine)
