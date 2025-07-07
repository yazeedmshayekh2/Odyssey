from typing import Dict, List, Optional, Union
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, HttpUrl
import shutil
import tempfile
import json
from car_verification import CarImageVerifier
from datetime import datetime
from database import get_db, CarReference, VerificationAttempt, deserialize_features, serialize_features
from sqlalchemy.orm import Session
from fastapi import Depends
from models import CarReferenceResponse, VerificationAttemptResponse, UploadResponse
import requests
import uuid

app = FastAPI(
    title="Car Verification API",
    description="API for verifying car images using YOLOv11 + InceptionV3 with stored embeddings",
    version="2.0.0"
)

# Create directories
os.makedirs("static", exist_ok=True)
os.makedirs("static/visualizations", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("uploads/reference", exist_ok=True)
os.makedirs("uploads/verification", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Templates
templates = Jinja2Templates(directory="templates")

# Allowed image extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Initialize the car verifier
verifier = CarImageVerifier()

def is_allowed_file(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

def save_uploaded_file(upload_file: UploadFile, directory: str, prefix: str = "") -> str:
    """Save uploaded file and return the file path"""
    if not is_allowed_file(upload_file.filename):
        raise HTTPException(status_code=400, detail=f"File type not allowed: {upload_file.filename}")
    
    # Generate unique filename
    file_extension = os.path.splitext(upload_file.filename)[1]
    unique_filename = f"{prefix}_{uuid.uuid4().hex}{file_extension}"
    file_path = os.path.join(directory, unique_filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return file_path

class ReferenceImageUrls(BaseModel):
    front_url: HttpUrl
    back_url: HttpUrl
    left_url: HttpUrl
    right_url: HttpUrl



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with upload forms"""
    return templates.TemplateResponse("index.html", {"request": request})

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

@app.post("/upload-reference", response_model=UploadResponse)
async def upload_reference_images(
    model: str = Form(...),
    year: int = Form(...),
    description: str = Form(None),
    front_image: UploadFile = File(...),
    back_image: UploadFile = File(...),
    left_image: UploadFile = File(...),
    right_image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload reference images for a car model"""
    
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
        
        # Save images
        reference_dir = f"uploads/reference/{model}_{year}"
        os.makedirs(reference_dir, exist_ok=True)
        
        front_path = save_uploaded_file(front_image, reference_dir, "front")
        back_path = save_uploaded_file(back_image, reference_dir, "back")
        left_path = save_uploaded_file(left_image, reference_dir, "left")
        right_path = save_uploaded_file(right_image, reference_dir, "right")
        
        # Extract features for each uploaded image
        print(f"🧠 Extracting features for {model} {year} reference images...")
        
        image_paths = {
            "front": front_path,
            "back": back_path,
            "left": left_path,
            "right": right_path
        }
        
        # Create visualization directory for reference images
        ref_vis_dir = f"static/visualizations/reference_{model}_{year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(ref_vis_dir, exist_ok=True)
        
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
                "front": front_path,
                "back": back_path,
                "left": left_path,
                "right": right_path
            }
        )
        
    except Exception as e:
        # Clean up any uploaded files in case of error
        for path in [front_path, back_path, left_path, right_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        raise HTTPException(status_code=500, detail=f"Error processing reference images: {str(e)}")

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

@app.post("/verify-car")
async def verify_car_images(
    model: str = Form(...),
    year: int = Form(...),
    front_image: UploadFile = File(...),
    back_image: UploadFile = File(...),
    left_image: UploadFile = File(...),
    right_image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Verify uploaded car images against reference images"""
    
    try:
        # Find reference images
        car_ref = db.query(CarReference).filter(
            CarReference.model == model,
            CarReference.year == year
        ).first()
        
        if not car_ref:
            raise HTTPException(
                status_code=404,
                detail=f"No reference images found for {model} {year}"
            )
        
        # Save uploaded images for verification
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        verification_dir = f"uploads/verification/{model}_{year}_{timestamp}"
        os.makedirs(verification_dir, exist_ok=True)
        
        # Create visualization directory
        vis_dir = f"static/visualizations/{model}_{year}_{timestamp}"
        os.makedirs(vis_dir, exist_ok=True)
        
        uploaded_front = save_uploaded_file(front_image, verification_dir, "front")
        uploaded_back = save_uploaded_file(back_image, verification_dir, "back")
        uploaded_left = save_uploaded_file(left_image, verification_dir, "left")
        uploaded_right = save_uploaded_file(right_image, verification_dir, "right")
        
        # Retrieve stored reference features from database
        print(f"🔍 Retrieving stored features for {model} {year}...")
        stored_reference_features = {}
        
        # Check if we have stored features (for backwards compatibility)
        if car_ref.front_features:
            stored_reference_features["front"] = deserialize_features(car_ref.front_features)
            stored_reference_features["back"] = deserialize_features(car_ref.back_features)
            stored_reference_features["left"] = deserialize_features(car_ref.left_features)
            stored_reference_features["right"] = deserialize_features(car_ref.right_features)
            
            print("✅ Using pre-computed reference features for faster verification")
            
            uploaded_images = {
                "front": uploaded_front,
                "back": uploaded_back,
                "left": uploaded_left,
                "right": uploaded_right
            }
            
            # Perform verification using stored features (much faster!)
            verification_results = verifier.verify_with_stored_features(stored_reference_features, uploaded_images, save_visualizations=True, vis_dir=vis_dir)
            
                else:
            # Fallback to old method if no stored features (backwards compatibility)
            print("⚠️ No stored features found, falling back to real-time feature extraction")
            
            reference_images = {
                "front": car_ref.front_image_path,
                "back": car_ref.back_image_path,
                "left": car_ref.left_image_path,
                "right": car_ref.right_image_path
            }
            
            uploaded_images = {
                "front": uploaded_front,
                "back": uploaded_back,
                "left": uploaded_left,
                "right": uploaded_right
            }
            
            # Perform verification using old method
            verification_results = verifier.verify_car_images(reference_images, uploaded_images)
        
        # Save verification attempt to database
        verification_attempt = VerificationAttempt(
            car_model=model,
            car_year=year,
            front_similarity=json.dumps(verification_results.get("front", {})),
            back_similarity=json.dumps(verification_results.get("back", {})),
            left_similarity=json.dumps(verification_results.get("left", {})),
            right_similarity=json.dumps(verification_results.get("right", {})),
            overall_result="MATCH" if verification_results["overall_result"]["is_same_car"] else "NO_MATCH",
            uploaded_front_path=uploaded_front,
            uploaded_back_path=uploaded_back,
            uploaded_left_path=uploaded_left,
            uploaded_right_path=uploaded_right
        )
        
        db.add(verification_attempt)
        db.commit()
        
        # Convert results to response format
        def create_image_comparison_result(data: dict) -> dict:
            cosine_sim = data.get("cosine_similarity", 0.0)
            result = {
                "feature_similarity": cosine_sim,  # Using cosine similarity as main feature similarity
                "histogram_similarity": cosine_sim,  # For compatibility with frontend
                "structural_similarity": cosine_sim,  # For compatibility with frontend
                "overall_similarity": cosine_sim,
                "is_match": data.get("is_match", False),
                "confidence": data.get("confidence", "Low"),
                "error": data.get("error"),
                # Include additional data from verification results
                "upload_car_detected": data.get("upload_car_detected", False),
                "upload_detection_confidence": data.get("upload_detection_confidence", 0.0),
                "upload_detection_visualization": data.get("upload_detection_visualization"),
                "upload_preprocessed_visualization": data.get("upload_preprocessed_visualization"),
                "model_used": data.get("model_used", ""),
                "processing_method": data.get("processing_method", "")
            }
            return result
        
        return {
            "verification_id": verification_attempt.id,
            "reference_car": CarReferenceResponse.from_orm(car_ref),
            "results": {
                "front": create_image_comparison_result(verification_results.get("front", {})),
                "back": create_image_comparison_result(verification_results.get("back", {})),
                "left": create_image_comparison_result(verification_results.get("left", {})),
                "right": create_image_comparison_result(verification_results.get("right", {})),
                "overall_result": verification_results["overall_result"]
            },
            "uploaded_images": uploaded_images
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during verification: {str(e)}")

@app.get("/references", response_model=List[CarReferenceResponse])
async def get_all_references(db: Session = Depends(get_db)):
    """Get all car references"""
    references = db.query(CarReference).all()
    return references

@app.get("/references/{model}/{year}", response_model=CarReferenceResponse)
async def get_reference(model: str, year: int, db: Session = Depends(get_db)):
    """Get specific car reference"""
    car_ref = db.query(CarReference).filter(
        CarReference.model == model,
        CarReference.year == year
    ).first()
    
    if not car_ref:
        raise HTTPException(status_code=404, detail="Car reference not found")
    
    return car_ref

@app.get("/verifications", response_model=List[VerificationAttemptResponse])
async def get_verification_history(db: Session = Depends(get_db)):
    """Get all verification attempts"""
    attempts = db.query(VerificationAttempt).order_by(VerificationAttempt.created_at.desc()).all()
    return attempts

@app.delete("/references/{model}/{year}")
async def delete_reference(model: str, year: int, db: Session = Depends(get_db)):
    """Delete car reference and associated files"""
    car_ref = db.query(CarReference).filter(
        CarReference.model == model,
        CarReference.year == year
    ).first()
    
    if not car_ref:
        raise HTTPException(status_code=404, detail="Car reference not found")
    
    # Delete files
    for image_path in [car_ref.front_image_path, car_ref.back_image_path, 
                      car_ref.left_image_path, car_ref.right_image_path]:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
    
    # Delete directory if empty
    reference_dir = f"uploads/reference/{model}_{year}"
    if os.path.exists(reference_dir):
        try:
            os.rmdir(reference_dir)
        except OSError:
            pass  # Directory not empty
    
    # Delete from database
    db.delete(car_ref)
    db.commit()
    
    return {"message": f"Reference for {model} {year} deleted successfully"}

# Simple health check endpoint
@app.get("/health")
def health_check():
    """Check if the API is running and models are loaded"""
    return {
        "status": "healthy",
        "models_loaded": verifier.feature_extractor.model is not None and verifier.yolo_model is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 