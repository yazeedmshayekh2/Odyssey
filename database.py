from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import json
import numpy as np

# Database URL - using SQLite for simplicity
SQLALCHEMY_DATABASE_URL = "sqlite:///./car_database.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class CarReference(Base):
    __tablename__ = "car_references"
    
    id = Column(Integer, primary_key=True, index=True)
    model = Column(String, index=True)
    year = Column(Integer, index=True)
    front_image_path = Column(String)
    back_image_path = Column(String)
    left_image_path = Column(String)
    right_image_path = Column(String)
    # Processed image paths - what the AI model actually sees
    front_processed_path = Column(String, nullable=True)
    back_processed_path = Column(String, nullable=True)
    left_processed_path = Column(String, nullable=True)
    right_processed_path = Column(String, nullable=True)
    # Detection visualization paths - showing bounding boxes
    front_detection_path = Column(String, nullable=True)
    back_detection_path = Column(String, nullable=True)
    left_detection_path = Column(String, nullable=True)
    right_detection_path = Column(String, nullable=True)
    # Feature vectors extracted using InceptionV3 + Advanced Preprocessing
    front_features = Column(Text)  # JSON serialized numpy array
    back_features = Column(Text)   # JSON serialized numpy array
    left_features = Column(Text)   # JSON serialized numpy array
    right_features = Column(Text)  # JSON serialized numpy array
    features_model = Column(String, default="InceptionV3")  # Model used for feature extraction
    features_version = Column(String, default="1.0")  # Version for feature compatibility
    created_at = Column(DateTime, default=datetime.utcnow)
    description = Column(Text, nullable=True)

class VerificationAttempt(Base):
    __tablename__ = "verification_attempts"
    
    id = Column(Integer, primary_key=True, index=True)
    car_model = Column(String)
    car_year = Column(Integer)
    front_similarity = Column(String)  # JSON string with similarity scores
    back_similarity = Column(String)
    left_similarity = Column(String)
    right_similarity = Column(String)
    overall_result = Column(String)  # "MATCH" or "NO_MATCH"
    created_at = Column(DateTime, default=datetime.utcnow)
    uploaded_front_path = Column(String)
    uploaded_back_path = Column(String)
    uploaded_left_path = Column(String)
    uploaded_right_path = Column(String)

# Helper functions for feature serialization
def serialize_features(features: np.ndarray) -> str:
    """Convert numpy array to JSON string for database storage"""
    if features is None:
        return None
    return json.dumps(features.tolist())

def deserialize_features(features_json: str) -> np.ndarray:
    """Convert JSON string back to numpy array"""
    if features_json is None:
        return None
    features_list = json.loads(features_json)
    return np.array(features_list, dtype=np.float32)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create upload directories
os.makedirs("uploads/reference", exist_ok=True)
os.makedirs("uploads/verification", exist_ok=True) 