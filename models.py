from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class CarReferenceCreate(BaseModel):
    model: str
    year: int
    description: Optional[str] = None

class CarReferenceResponse(BaseModel):
    id: int
    model: str
    year: int
    front_image_path: Optional[str]
    back_image_path: Optional[str]
    left_image_path: Optional[str]
    right_image_path: Optional[str]
    # Processed images that are actually fed to the AI model
    front_processed_path: Optional[str] = None
    back_processed_path: Optional[str] = None
    left_processed_path: Optional[str] = None
    right_processed_path: Optional[str] = None
    # Detection visualization images showing bounding boxes
    front_detection_path: Optional[str] = None
    back_detection_path: Optional[str] = None
    left_detection_path: Optional[str] = None
    right_detection_path: Optional[str] = None
    created_at: datetime
    description: Optional[str]
    
    class Config:
        from_attributes = True

class CarVerificationRequest(BaseModel):
    model: str
    year: int

class ImageComparisonResult(BaseModel):
    feature_similarity: float
    histogram_similarity: float
    structural_similarity: float
    overall_similarity: float
    is_match: bool
    confidence: str
    error: Optional[str] = None

class CarVerificationResult(BaseModel):
    front: ImageComparisonResult
    back: ImageComparisonResult
    left: ImageComparisonResult
    right: ImageComparisonResult
    overall_result: Dict[str, Any]
    reference_car: CarReferenceResponse

class VerificationAttemptResponse(BaseModel):
    id: int
    car_model: str
    car_year: int
    overall_result: str
    created_at: datetime
    front_similarity: Optional[str]
    back_similarity: Optional[str]
    left_similarity: Optional[str]
    right_similarity: Optional[str]
    
    class Config:
        from_attributes = True

class UploadResponse(BaseModel):
    message: str
    car_id: Optional[int] = None
    uploaded_files: Dict[str, str] 