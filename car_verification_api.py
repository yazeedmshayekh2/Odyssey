from typing import Dict, List, Optional, Union
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel, HttpUrl
import shutil
import tempfile
from car_verification import CarImageVerifier
from datetime import datetime
from database import get_db, CarReference, deserialize_features, serialize_features
from sqlalchemy.orm import Session
from fastapi import Depends
import requests
import uuid

app = FastAPI(
    title="Car Verification API",
    description="API for verifying car images using YOLOv11 + InceptionV3 with stored embeddings",
    version="2.0.0"
)

# Initialize the car verifier
verifier = CarImageVerifier()

class VerificationResult(BaseModel):
    side: str
    is_match: bool
    confidence: str
    similarity_score: float
    error: Optional[str] = None

class VerificationResponse(BaseModel):
    front_result: VerificationResult
    back_result: VerificationResult
    left_result: VerificationResult
    right_result: VerificationResult
    overall_match: bool
    average_similarity: float
    overall_confidence: str

class UploadResponse(BaseModel):
    message: str
    car_id: Optional[int] = None
    uploaded_files: Dict[str, str]

class ReferenceImageUrls(BaseModel):
    front_url: HttpUrl
    back_url: HttpUrl
    left_url: HttpUrl
    right_url: HttpUrl

def save_upload_file_tmp(upload_file: UploadFile) -> str:
    """Save an upload file to a temporary file and return the path"""
    try:
        suffix = os.path.splitext(upload_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            return tmp.name
    finally:
        upload_file.file.close()

def cleanup_temp_files(file_paths: List[str]):
    """Clean up temporary files"""
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception as e:
            print(f"Error cleaning up {path}: {e}")

def download_image_to_temp(url: str) -> str:
    """Download image from URL to a temporary file"""
    try:
        # Create temp file with .jpg extension
        temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(temp_fd)

        # Download image
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Save to temp file
        with open(temp_path, 'wb') as f:
            f.write(response.content)

        return temp_path
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")

@app.post("/upload-reference-urls", response_model=UploadResponse)
async def upload_reference_urls(
    model: str = Form(...),
    year: int = Form(...),
    description: str = Form(None),
    front_url: str = Form(...),
    back_url: str = Form(...),
    left_url: str = Form(...),
    right_url: str = Form(...),
    db: Session = Depends(get_db)
):
    """Upload reference images from URLs and store their embeddings"""
    try:
        # Check if reference already exists
        existing_ref = db.query(CarReference).filter(
            CarReference.model == model, 
            CarReference.year == year
        ).first()
        
        if existing_ref:
            raise HTTPException(
                status_code=400, 
                detail=f"Reference images for {model} {year} already exist"
            )
        
        # Create directories
        reference_dir = f"uploads/reference/{model}_{year}"
        os.makedirs(reference_dir, exist_ok=True)
        
        # Create visualization directory for reference images
        ref_vis_dir = f"static/visualizations/reference_{model}_{year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(ref_vis_dir, exist_ok=True)

        # Download and save images
        front_path = os.path.join(reference_dir, f"front_{uuid.uuid4().hex}.jpg")
        back_path = os.path.join(reference_dir, f"back_{uuid.uuid4().hex}.jpg")
        left_path = os.path.join(reference_dir, f"left_{uuid.uuid4().hex}.jpg")
        right_path = os.path.join(reference_dir, f"right_{uuid.uuid4().hex}.jpg")

        # Download images
        response = requests.get(front_url, timeout=10)
        response.raise_for_status()
        with open(front_path, 'wb') as f:
            f.write(response.content)

        response = requests.get(back_url, timeout=10)
        response.raise_for_status()
        with open(back_path, 'wb') as f:
            f.write(response.content)

        response = requests.get(left_url, timeout=10)
        response.raise_for_status()
        with open(left_path, 'wb') as f:
            f.write(response.content)

        response = requests.get(right_url, timeout=10)
        response.raise_for_status()
        with open(right_path, 'wb') as f:
            f.write(response.content)

        # Process images and extract features
        print(f"🧠 Extracting features for {model} {year} reference images...")
        
        image_paths = {
            "front": front_path,
            "back": back_path,
            "left": left_path,
            "right": right_path
        }
        
        extracted_features = {}
        processed_paths = {}
        detection_paths = {}
        
        for side, path in image_paths.items():
            print(f"🔍 Processing {side} image...")
            # Apply enhanced preprocessing for reference images
            result = verifier.process_image(path, save_visualizations=True, vis_dir=ref_vis_dir, skip_preprocessing=False, is_reference=True, view_type=side)
            if result['features'] is not None:
                extracted_features[side] = serialize_features(result['features'])
                print(f"✅ {side} features extracted and serialized with enhanced preprocessing")
                
                # Store paths to processed visualizations
                processed_paths[side] = result.get('preprocessed_visualization')
                detection_paths[side] = result.get('detection_visualization')
                
                print(f"📷 Processed image saved: {processed_paths[side]}")
                print(f"🎯 Detection image saved: {detection_paths[side]}")
            else:
                print(f"⚠️ Failed to extract features for {side} image")
                extracted_features[side] = None
                processed_paths[side] = None
                detection_paths[side] = None
        
        print(f"🎯 Feature extraction completed for {model} {year}")
        
        # Create database entry with features and processed image paths
        car_ref = CarReference(
            model=model,
            year=year,
            description=description,
            front_image_path=front_path,
            back_image_path=back_path,
            left_image_path=left_path,
            right_image_path=right_path,
            # Store processed image paths
            front_processed_path=processed_paths.get("front"),
            back_processed_path=processed_paths.get("back"),
            left_processed_path=processed_paths.get("left"),
            right_processed_path=processed_paths.get("right"),
            # Store detection visualization paths
            front_detection_path=detection_paths.get("front"),
            back_detection_path=detection_paths.get("back"),
            left_detection_path=detection_paths.get("left"),
            right_detection_path=detection_paths.get("right"),
            front_features=extracted_features["front"],
            back_features=extracted_features["back"],
            left_features=extracted_features["left"],
            right_features=extracted_features["right"],
            features_model="InceptionV3",
            features_version="1.0"
        )
        
        db.add(car_ref)
        db.commit()
        db.refresh(car_ref)
        
        return UploadResponse(
            message=f"Reference images for {model} {year} uploaded successfully",
            car_id=car_ref.id,
            uploaded_files={
                "front": front_url,
                "back": back_url,
                "left": left_url,
                "right": right_url
            }
        )
        
    except Exception as e:
        # Clean up any downloaded files in case of error
        for path in [front_path, back_path, left_path, right_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        raise HTTPException(status_code=500, detail=f"Error processing reference images: {str(e)}")

@app.post("/verify/{model}/{year}", response_model=VerificationResponse)
async def verify_car_images(
    model: str,
    year: int,
    upload_front: UploadFile = File(...),
    upload_back: UploadFile = File(...),
    upload_left: UploadFile = File(...),
    upload_right: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Verify car images by comparing uploaded images with stored embeddings.
    
    Parameters:
    - model: Car model to verify against
    - year: Car year to verify against
    - upload_front, upload_back, upload_left, upload_right: Images to verify
    
    Returns:
    - Verification results for each side and overall match status
    """
    temp_files = []
    
    try:
        # Get stored embeddings from database
        car_ref = db.query(CarReference).filter(
            CarReference.model == model,
            CarReference.year == year
        ).first()
        
        if not car_ref:
            raise HTTPException(
                status_code=404,
                detail=f"No reference embeddings found for {model} {year}"
            )
        
        # Get stored embeddings
        stored_embeddings = {
            "front": deserialize_features(car_ref.front_features),
            "back": deserialize_features(car_ref.back_features),
            "left": deserialize_features(car_ref.left_features),
            "right": deserialize_features(car_ref.right_features)
        }
        
        # Save uploaded files temporarily
        upload_paths = {
            "front": save_upload_file_tmp(upload_front),
            "back": save_upload_file_tmp(upload_back),
            "left": save_upload_file_tmp(upload_left),
            "right": save_upload_file_tmp(upload_right)
        }
        temp_files.extend(upload_paths.values())
        
        # Process each uploaded image and compare with stored embeddings
        results = {}
        total_similarity = 0
        matches = 0
        
        for side, path in upload_paths.items():
            # Process uploaded image
            processed = verifier.process_image(path)
            
            if processed['features'] is not None:
                # Compare with stored embedding
                similarity = verifier.calculate_cosine_similarity(
                    processed['features'],
                    stored_embeddings[side]
                )
                
                # Determine match based on side-specific thresholds
                threshold = 0.75 if side == "back" else 0.80
                is_match = similarity >= threshold
                
                # Determine confidence level
                if similarity >= 0.95:
                    confidence = "Very High"
                elif similarity >= 0.85:
                    confidence = "High"
                elif similarity >= 0.75:
                    confidence = "Medium"
                else:
                    confidence = "Low"
                
                results[side] = {
                    "is_match": is_match,
                    "confidence": confidence,
                    "cosine_similarity": similarity
                }
                
                total_similarity += similarity
                if is_match:
                    matches += 1
            else:
                results[side] = {
                    "is_match": False,
                    "confidence": "Low",
                    "cosine_similarity": 0.0,
                    "error": "Feature extraction failed"
                }
        
        # Calculate overall results
        average_similarity = total_similarity / 4
        overall_match = matches >= 3  # At least 3 sides must match
        
        if average_similarity >= 0.95:
            overall_confidence = "Very High"
        elif average_similarity >= 0.85:
            overall_confidence = "High"
        elif average_similarity >= 0.75:
            overall_confidence = "Medium"
        else:
            overall_confidence = "Low"
        
        # Format response
        response = VerificationResponse(
            front_result=VerificationResult(
                side="front",
                is_match=results["front"]["is_match"],
                confidence=results["front"]["confidence"],
                similarity_score=results["front"]["cosine_similarity"],
                error=results["front"].get("error")
            ),
            back_result=VerificationResult(
                side="back",
                is_match=results["back"]["is_match"],
                confidence=results["back"]["confidence"],
                similarity_score=results["back"]["cosine_similarity"],
                error=results["back"].get("error")
            ),
            left_result=VerificationResult(
                side="left",
                is_match=results["left"]["is_match"],
                confidence=results["left"]["confidence"],
                similarity_score=results["left"]["cosine_similarity"],
                error=results["left"].get("error")
            ),
            right_result=VerificationResult(
                side="right",
                is_match=results["right"]["is_match"],
                confidence=results["right"]["confidence"],
                similarity_score=results["right"]["cosine_similarity"],
                error=results["right"].get("error")
            ),
            overall_match=overall_match,
            average_similarity=average_similarity,
            overall_confidence=overall_confidence
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up temporary files
        cleanup_temp_files(temp_files)

# Simple health check endpoint
@app.get("/health")
def health_check():
    """Check if the API is running and models are loaded"""
    return {
        "status": "healthy",
        "models_loaded": verifier.inception_model is not None and verifier.yolo_model is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 