from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List, Dict
import shutil
import os
import json
import uuid
from datetime import datetime
import base64
from io import BytesIO
import zipfile
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import tempfile
from PIL import Image as PILImage
import asyncio
from collections import deque
import queue
import threading
import requests

from database import get_db, CarReference, VerificationAttempt, serialize_features, deserialize_features
from models import (
    CarReferenceCreate, CarReferenceResponse, CarVerificationRequest,
    CarVerificationResult, VerificationAttemptResponse, UploadResponse,
    ImageComparisonResult
)

app = FastAPI(title="Car Verification System", version="1.0.0")

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

def prepare_image_for_pdf(image_path: str, max_width: float = 1.5*inch, max_height: float = 1.5*inch) -> ReportLabImage:
    """Prepare an image for inclusion in PDF with proper sizing"""
    try:
        if not os.path.exists(image_path):
            return None
            
        # Open and resize image if needed
        with PILImage.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate aspect ratio preserving dimensions
            original_width, original_height = img.size
            aspect_ratio = original_width / original_height
            
            if original_width > original_height:
                width = min(max_width, max_width)
                height = width / aspect_ratio
                if height > max_height:
                    height = max_height
                    width = height * aspect_ratio
            else:
                height = min(max_height, max_height)
                width = height * aspect_ratio
                if width > max_width:
                    width = max_width
                    height = width / aspect_ratio
            
            # Create ReportLab Image object
            return ReportLabImage(image_path, width=width, height=height)
            
    except Exception as e:
        print(f"Error preparing image {image_path}: {e}")
        return None

# Create a queue for bulk upload status updates
bulk_upload_status_queue = queue.Queue()

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and ngrok testing"""
    return {
        "status": "healthy",
        "service": "Car Verification API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get('/bulk-upload-status')
async def bulk_upload_status():
    """Server-Sent Events endpoint for bulk upload status"""
    async def event_generator():
        while True:
            try:
                # Non-blocking check for new status updates
                try:
                    status = bulk_upload_status_queue.get_nowait()
                    yield f"data: {json.dumps(status)}\n\n"
                except queue.Empty:
                    await asyncio.sleep(0.1)  # Small delay to prevent CPU spinning
            except Exception as e:
                print(f"Error in SSE stream: {e}")
                break
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/bulk-upload")
async def start_bulk_upload(background_tasks: BackgroundTasks, 
                           batch_size: int = 10, 
                           start_idx: int = 0):
    """Start bulk upload process in the background"""
    try:
        # Import the bulk uploader
        from bulk_upload_imagin import BulkImageUploader
        
        def status_callback(status_data):
            """Callback to receive status updates from BulkImageUploader"""
            bulk_upload_status_queue.put(status_data)
        
        def run_bulk_upload():
            uploader = BulkImageUploader()
            uploader.run_bulk_upload(
                excel_file="odysseydatasetcarimages.xlsx",
                start_idx=start_idx,
                batch_size=batch_size,
                status_callback=status_callback
            )
        
        # Start the bulk upload in background
        background_tasks.add_task(run_bulk_upload)
        
        return {
            "message": "Bulk upload started in background",
            "batch_size": batch_size,
            "start_index": start_idx,
            "status": "started"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start bulk upload: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with upload forms"""
    return templates.TemplateResponse("index.html", {"request": request})

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
            result = verifier.process_image(path, save_visualizations=True, vis_dir=ref_vis_dir, skip_preprocessing=False, is_reference=True)
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
        raise HTTPException(status_code=500, detail=f"Error uploading reference images: {str(e)}")

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
        
        # Apply automatic rejection logic for low confidence
        overall_result = verification_results["overall_result"]
        weighted_similarity = overall_result.get("weighted_similarity", 0.0)
        
        # Auto-reject if weighted similarity is below 80% (Low confidence)
        if weighted_similarity < 0.80:
            final_result = "REJECTED"
            overall_result["is_same_car"] = False
            overall_result["auto_rejected"] = True
            overall_result["rejection_reason"] = "Low verification accuracy - automatic rejection for reliability"
            print(f"🚫 Auto-rejecting car: {weighted_similarity:.1%} similarity is below 80% threshold")
        else:
            final_result = "MATCH" if overall_result["is_same_car"] else "NO_MATCH"
            overall_result["auto_rejected"] = False
        
        # Save verification attempt to database
        verification_attempt = VerificationAttempt(
            car_model=model,
            car_year=year,
            front_similarity=json.dumps(verification_results.get("front", {})),
            back_similarity=json.dumps(verification_results.get("back", {})),
            left_similarity=json.dumps(verification_results.get("left", {})),
            right_similarity=json.dumps(verification_results.get("right", {})),
            overall_result=final_result,
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

@app.delete("/reference/{reference_id}")
async def delete_reference_by_id(reference_id: int, db: Session = Depends(get_db)):
    """Delete a car reference by ID with comprehensive cleanup"""
    reference = db.query(CarReference).filter(CarReference.id == reference_id).first()
    if not reference:
        raise HTTPException(status_code=404, detail="Reference not found")
    
    # Get car info for response
    car_info = f"{reference.model} {reference.year}"
    
    # Delete all associated files
    file_paths = [
        reference.front_image_path,
        reference.back_image_path,
        reference.left_image_path,
        reference.right_image_path,
        reference.front_processed_path,
        reference.back_processed_path,
        reference.left_processed_path,
        reference.right_processed_path,
        reference.front_detection_path,
        reference.back_detection_path,
        reference.left_detection_path,
        reference.right_detection_path
    ]
    
    deleted_files = []
    failed_files = []
    
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                deleted_files.append(file_path)
            except Exception as e:
                failed_files.append(f"{file_path}: {str(e)}")
                print(f"Warning: Could not delete file {file_path}: {e}")
    
    # Try to delete the car directory if it's empty
    if reference.front_image_path:
        car_dir = os.path.dirname(reference.front_image_path)
        try:
            if os.path.exists(car_dir) and len(os.listdir(car_dir)) == 0:
                os.rmdir(car_dir)
                deleted_files.append(f"Directory: {car_dir}")
        except Exception as e:
            print(f"Warning: Could not delete directory {car_dir}: {e}")
    
    # Try to delete visualization directories
    try:
        import glob
        import shutil
        vis_pattern = f"static/visualizations/reference_{reference.model}_{reference.year}_*"
        vis_dirs = glob.glob(vis_pattern)
        for vis_dir in vis_dirs:
            if os.path.isdir(vis_dir):
                try:
                    shutil.rmtree(vis_dir)
                    deleted_files.append(f"Visualization: {vis_dir}")
                except Exception as e:
                    failed_files.append(f"Visualization {vis_dir}: {str(e)}")
    except Exception as e:
        print(f"Warning: Error during visualization cleanup: {e}")
    
    # Delete from database
    db.delete(reference)
    db.commit()
    
    return {
        "message": f"Reference '{car_info}' deleted successfully",
        "deleted_files": len(deleted_files),
        "failed_files": len(failed_files),
        "details": {
            "deleted": deleted_files[:10],  # Limit to first 10 for response size
            "failed": failed_files
        }
    }

@app.get("/list-test-cases")
async def list_test_cases(db: Session = Depends(get_db)):
    """List all verification attempts (test cases) with basic information"""
    try:
        attempts = db.query(VerificationAttempt).order_by(VerificationAttempt.created_at.desc()).all()
        
        # Convert to simplified format for frontend
        test_cases = []
        for attempt in attempts:
            try:
                # Parse similarity data
                front_data = json.loads(attempt.front_similarity) if attempt.front_similarity else {}
                back_data = json.loads(attempt.back_similarity) if attempt.back_similarity else {}
                left_data = json.loads(attempt.left_similarity) if attempt.left_similarity else {}
                right_data = json.loads(attempt.right_similarity) if attempt.right_similarity else {}
                
                # Calculate average similarity
                similarities = [
                    front_data.get('cosine_similarity', 0),
                    back_data.get('cosine_similarity', 0),
                    left_data.get('cosine_similarity', 0),
                    right_data.get('cosine_similarity', 0)
                ]
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0
                
                test_cases.append({
                    "id": attempt.id,
                    "car_model": attempt.car_model,
                    "car_year": attempt.car_year,
                    "overall_result": attempt.overall_result,
                    "average_similarity": round(avg_similarity, 4),
                    "created_at": attempt.created_at.isoformat(),
                    "test_date": attempt.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "confidence": "High" if avg_similarity >= 0.8 else "Medium" if avg_similarity >= 0.7 else "Low"
                })
            except Exception as e:
                print(f"Error processing attempt {attempt.id}: {e}")
                continue
        
        return {"test_cases": test_cases}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving test cases: {str(e)}")

@app.post("/download-selected-test-cases")
async def download_selected_test_cases(
    case_ids: str = Form(...),  # Comma-separated list of case IDs
    include_images: bool = Form(default=True),
    db: Session = Depends(get_db)
):
    """Download selected test cases as a comprehensive PDF report"""
    try:
        # Parse case IDs
        try:
            selected_ids = [int(id.strip()) for id in case_ids.split(',') if id.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid case ID format")
        
        if not selected_ids:
            raise HTTPException(status_code=400, detail="No test cases selected")
        
        # Retrieve selected test cases
        test_cases = db.query(VerificationAttempt).filter(
            VerificationAttempt.id.in_(selected_ids)
        ).order_by(VerificationAttempt.created_at.desc()).all()
        
        if not test_cases:
            raise HTTPException(status_code=404, detail="No test cases found for the selected IDs")
        
        # Create temporary file for PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            pdf_path = temp_file.name
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Add title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("Car Verification Test Cases Report", title_style))
        story.append(Spacer(1, 20))
        
        # Add report metadata
        metadata_style = ParagraphStyle('Metadata', parent=styles['Normal'], fontSize=10)
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", metadata_style))
        story.append(Paragraph(f"Total test cases: {len(test_cases)}", metadata_style))
        story.append(Spacer(1, 20))
        
        # Process each test case
        for i, test_case in enumerate(test_cases):
            try:
                # Parse similarity data
                front_data = json.loads(test_case.front_similarity) if test_case.front_similarity else {}
                back_data = json.loads(test_case.back_similarity) if test_case.back_similarity else {}
                left_data = json.loads(test_case.left_similarity) if test_case.left_similarity else {}
                right_data = json.loads(test_case.right_similarity) if test_case.right_similarity else {}
                
                # Calculate statistics
                similarities = [
                    front_data.get('cosine_similarity', 0),
                    back_data.get('cosine_similarity', 0),
                    left_data.get('cosine_similarity', 0),
                    right_data.get('cosine_similarity', 0)
                ]
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0
                
                # Add test case header
                case_style = ParagraphStyle('CaseHeader', parent=styles['Heading2'], fontSize=16, spaceAfter=10)
                story.append(Paragraph(f"Test Case #{test_case.id}: {test_case.car_model} {test_case.car_year}", case_style))
                
                # Add basic information table
                basic_info = [
                    ['Test Date:', test_case.created_at.strftime('%Y-%m-%d %H:%M:%S')],
                    ['Overall Result:', test_case.overall_result],
                    ['Average Similarity:', f"{avg_similarity:.1%}"],
                    ['Confidence:', "High" if avg_similarity >= 0.8 else "Medium" if avg_similarity >= 0.7 else "Low"]
                ]
                
                basic_table = Table(basic_info, colWidths=[2*inch, 3*inch])
                basic_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(basic_table)
                story.append(Spacer(1, 15))
                
                # Add detailed similarity results
                detail_style = ParagraphStyle('DetailHeader', parent=styles['Heading3'], fontSize=12)
                story.append(Paragraph("Detailed Similarity Results", detail_style))
                
                similarity_data = [
                    ['View', 'Similarity Score', 'Match Status', 'Confidence', 'Car Detected'],
                    [
                        'Front',
                        f"{front_data.get('cosine_similarity', 0):.1%}",
                        "✓ Match" if front_data.get('is_match', False) else "✗ No Match",
                        front_data.get('confidence', 'Unknown'),
                        "✓ Yes" if front_data.get('upload_car_detected', False) else "✗ No"
                    ],
                    [
                        'Back',
                        f"{back_data.get('cosine_similarity', 0):.1%}",
                        "✓ Match" if back_data.get('is_match', False) else "✗ No Match",
                        back_data.get('confidence', 'Unknown'),
                        "✓ Yes" if back_data.get('upload_car_detected', False) else "✗ No"
                    ],
                    [
                        'Left',
                        f"{left_data.get('cosine_similarity', 0):.1%}",
                        "✓ Match" if left_data.get('is_match', False) else "✗ No Match",
                        left_data.get('confidence', 'Unknown'),
                        "✓ Yes" if left_data.get('upload_car_detected', False) else "✗ No"
                    ],
                    [
                        'Right',
                        f"{right_data.get('cosine_similarity', 0):.1%}",
                        "✓ Match" if right_data.get('is_match', False) else "✗ No Match",
                        right_data.get('confidence', 'Unknown'),
                        "✓ Yes" if right_data.get('upload_car_detected', False) else "✗ No"
                    ]
                ]
                
                similarity_table = Table(similarity_data, colWidths=[1*inch, 1.2*inch, 1.2*inch, 1*inch, 1.1*inch])
                similarity_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
                ]))
                story.append(similarity_table)
                
                # Add images section if include_images is True
                if include_images:
                    story.append(Spacer(1, 15))
                    story.append(Paragraph("Visual Comparison", detail_style))
                    
                    # Get reference images for this car model/year
                    reference_car = db.query(CarReference).filter(
                        CarReference.model == test_case.car_model,
                        CarReference.year == test_case.car_year
                    ).first()
                    
                    if reference_car:
                        # Create image comparison tables for each view
                        views = ['front', 'back', 'left', 'right']
                        for view in views:
                            # Get uploaded image path
                            uploaded_path = getattr(test_case, f'uploaded_{view}_path', None)
                            # Get reference image path  
                            reference_path = getattr(reference_car, f'{view}_image_path', None)
                            
                            if uploaded_path and reference_path:
                                # Prepare images
                                uploaded_img = prepare_image_for_pdf(uploaded_path)
                                reference_img = prepare_image_for_pdf(reference_path)
                                
                                if uploaded_img and reference_img:
                                    # Create comparison table for this view
                                    view_title = f"{view.title()} View Comparison"
                                    story.append(Spacer(1, 10))
                                    story.append(Paragraph(view_title, ParagraphStyle('ViewTitle', parent=styles['Normal'], fontSize=11, fontName='Helvetica-Bold')))
                                    story.append(Spacer(1, 8))
                                    
                                    # Get similarity data for this view
                                    view_data = locals()[f'{view}_data']
                                    similarity_text = f"Similarity: {view_data.get('cosine_similarity', 0):.1%}"
                                    match_text = "✓ Match" if view_data.get('is_match', False) else "✗ No Match"
                                    
                                    image_table_data = [
                                        ['Uploaded Image', 'Reference Image'],
                                        [uploaded_img, reference_img],
                                        [similarity_text, match_text]
                                    ]
                                    
                                    image_table = Table(image_table_data, colWidths=[2.75*inch, 2.75*inch])
                                    image_table.setStyle(TableStyle([
                                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                        ('VALIGN', (0, 1), (-1, 1), 'MIDDLE'),
                                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                        ('FONTNAME', (0, 2), (-1, 2), 'Helvetica'),
                                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                                        ('TOPPADDING', (0, 1), (-1, 1), 10),
                                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                                    ]))
                                    story.append(image_table)
                    
                    story.append(Spacer(1, 15))
                
                # Add technical details
                if front_data.get('model_used'):
                    story.append(Paragraph("Technical Details", detail_style))
                    tech_info = [
                        ['AI Model:', front_data.get('model_used', 'N/A')],
                        ['Processing Method:', str(front_data.get('processing_method', 'N/A'))],
                    ]
                    tech_table = Table(tech_info, colWidths=[2*inch, 4*inch])
                    tech_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(tech_table)
                
                # Add page break between test cases (except for the last one)
                if i < len(test_cases) - 1:
                    story.append(PageBreak())
                else:
                    story.append(Spacer(1, 30))
                    
            except Exception as e:
                print(f"Error processing test case {test_case.id}: {e}")
                continue
        
        # Add summary section
        story.append(Paragraph("Summary", title_style))
        total_matches = sum(1 for tc in test_cases if tc.overall_result == "MATCH")
        total_cases = len(test_cases)
        match_rate = (total_matches / total_cases * 100) if total_cases > 0 else 0
        
        summary_data = [
            ['Total Test Cases:', str(total_cases)],
            ['Successful Matches:', str(total_matches)],
            ['Failed Matches:', str(total_cases - total_matches)],
            ['Match Rate:', f"{match_rate:.1f}%"]
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgreen),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 2, colors.black)
        ]))
        story.append(summary_table)
        
        # Build PDF
        doc.build(story)
        
        # Return the PDF file
        filename = f"test_cases_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return FileResponse(
            pdf_path,
            filename=filename,
            media_type="application/pdf",
            background=None  # Don't delete the file immediately
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

# Clean up the previous incorrect implementations
@app.post("/list-test-cases-by-car")
async def list_test_cases_by_car(
    model: str = Form(...),
    year: int = Form(...),
    db: Session = Depends(get_db)
):
    """List all test cases for a specific car model and year"""
    try:
        attempts = db.query(VerificationAttempt).filter(
            VerificationAttempt.car_model == model,
            VerificationAttempt.car_year == year
        ).order_by(VerificationAttempt.created_at.desc()).all()
        
        # Convert to simplified format for frontend
        test_cases = []
        for attempt in attempts:
            try:
                # Parse similarity data
                front_data = json.loads(attempt.front_similarity) if attempt.front_similarity else {}
                avg_similarity = front_data.get('cosine_similarity', 0)
                
                test_cases.append({
                    "id": attempt.id,
                    "overall_result": attempt.overall_result,
                    "average_similarity": round(avg_similarity, 4),
                    "created_at": attempt.created_at.isoformat(),
                    "test_date": attempt.created_at.strftime("%Y-%m-%d %H:%M:%S")
                })
            except Exception as e:
                print(f"Error processing attempt {attempt.id}: {e}")
                continue
        
        return {"test_cases": test_cases}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving test cases: {str(e)}")

@app.delete("/delete-test-case/{case_id}")
async def delete_test_case(case_id: int, db: Session = Depends(get_db)):
    """Delete a specific test case"""
    try:
        # Find the test case
        test_case = db.query(VerificationAttempt).filter(
            VerificationAttempt.id == case_id
        ).first()
        
        if not test_case:
            raise HTTPException(status_code=404, detail="Test case not found")
        
        # Get the uploaded file paths to clean up
        uploaded_files = [
            test_case.uploaded_front_path,
            test_case.uploaded_back_path,
            test_case.uploaded_left_path,
            test_case.uploaded_right_path
        ]
        
        # Delete the database record
        db.delete(test_case)
        db.commit()
        
        # Clean up uploaded files
        for file_path in uploaded_files:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"✅ Deleted file: {file_path}")
                except Exception as e:
                    print(f"⚠️ Could not delete file {file_path}: {e}")
        
        # Try to clean up visualization directory if it exists
        try:
            # Extract model and year from the test case to find visualization directory
            vis_pattern = f"static/visualizations/{test_case.car_model}_{test_case.car_year}_*"
            import glob
            vis_dirs = glob.glob(vis_pattern)
            for vis_dir in vis_dirs:
                if os.path.isdir(vis_dir):
                    # Check if directory contains files related to this test case
                    # This is a simple cleanup - you might want to be more specific
                    try:
                        import shutil
                        # Only remove if directory is empty or contains only related files
                        files_in_dir = os.listdir(vis_dir)
                        if len(files_in_dir) <= 8:  # Typically 8 files per verification (4 views x 2 types)
                            shutil.rmtree(vis_dir)
                            print(f"✅ Deleted visualization directory: {vis_dir}")
                    except Exception as e:
                        print(f"⚠️ Could not delete visualization directory {vis_dir}: {e}")
        except Exception as e:
            print(f"⚠️ Error during visualization cleanup: {e}")
        
        return {"message": f"Test case {case_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting test case: {str(e)}")

@app.post("/verify-with-studio-images")
async def verify_with_studio_images(
    uploaded_front: UploadFile = File(...),
    uploaded_back: UploadFile = File(...),
    uploaded_left: UploadFile = File(...),
    uploaded_right: UploadFile = File(...),
    studio_front_url: str = Form(...),
    studio_back_url: str = Form(...),
    studio_left_url: str = Form(...),
    studio_right_url: str = Form(...),
    plate_number: str = Form(...),
    color: str = Form(...)
):
    """Compare uploaded images to studio images and return similarity results, with plate number and color."""
    try:
        # Helper to download image from URL
        def download_image(url, save_dir, prefix):
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                raise Exception(f"Failed to download image from {url}")
            ext = os.path.splitext(url)[1] or ".jpg"
            filename = f"{prefix}_{uuid.uuid4().hex}{ext}"
            path = os.path.join(save_dir, filename)
            with open(path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return path

        # Create temp dir for studio images
        temp_dir = tempfile.mkdtemp(prefix="studio_compare_")
        try:
            # Save uploaded images
            uploaded_paths = {}
            for side, file in zip(["front", "back", "left", "right"], [uploaded_front, uploaded_back, uploaded_left, uploaded_right]):
                uploaded_paths[side] = save_uploaded_file(file, temp_dir, f"uploaded_{side}")

            # Download studio images
            studio_urls = {
                "front": studio_front_url,
                "back": studio_back_url,
                "left": studio_left_url,
                "right": studio_right_url
            }
            studio_paths = {}
            for side, url in studio_urls.items():
                studio_paths[side] = download_image(url, temp_dir, f"studio_{side}")

            # Compare each pair (uploaded vs studio)
            results = {}
            for side in ["front", "back", "left", "right"]:
                # Use the same logic as in verify_car_images, but for a single pair
                pair_result = verifier.verify_pair(uploaded_paths[side], studio_paths[side])
                results[side] = {
                    "cosine_similarity": pair_result.get("cosine_similarity", 0.0),
                    "is_match": pair_result.get("is_match", False),
                    "confidence": pair_result.get("confidence", "Unknown"),
                    "error": pair_result.get("error")
                }

            # Calculate overall similarity (average)
            similarities = [results[side]["cosine_similarity"] for side in ["front", "back", "left", "right"]]
            avg_similarity = sum(similarities) / len(similarities)

            return {
                "plate_number": plate_number,
                "color": color,
                "results": results,
                "average_similarity": avg_similarity
            }
        finally:
            # Clean up temp files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    import ssl
    # Replace with the actual paths to your certificate and private key
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER) 
    context.load_cert_chain('cert.pem', 'key.pem')
    uvicorn.run(app, host="0.0.0.0", port=443, ssl_context=context)