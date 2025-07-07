#!/usr/bin/env python3
"""
Bulk Upload Script for IMAGIN.studio Car Images
Reads the Excel file and downloads car images from IMAGIN.studio API
"""

import pandas as pd
import requests
import os
import time
import sqlite3
from datetime import datetime
from PIL import Image
from io import BytesIO
import uuid
import shutil
from database import get_db, CarReference, serialize_features
from car_verification import CarImageVerifier
from sqlalchemy.orm import Session

class BulkImageUploader:
    def __init__(self):
        self.verifier = CarImageVerifier()
        self.session_requests = requests.Session()
        self.session_requests.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.status_callback = None
        
        # Create directories
        os.makedirs("uploads/reference", exist_ok=True)
        os.makedirs("static/visualizations", exist_ok=True)
        
    def send_status(self, status_data):
        """Send status update via callback if available"""
        if self.status_callback:
            self.status_callback(status_data)
    
    def download_image(self, url: str, save_path: str, max_retries: int = 3) -> bool:
        """Download image from URL with retries"""
        for attempt in range(max_retries):
            try:
                print(f"📥 Downloading: {url}")
                response = self.session_requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    # Verify it's actually an image
                    img = Image.open(BytesIO(response.content))
                    
                    # Save the image
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    
                    print(f"✅ Downloaded: {save_path}")
                    return True
                else:
                    print(f"⚠️ HTTP {response.status_code}: {url}")
                    
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                    
        return False
    
    def process_car_batch(self, cars_df: pd.DataFrame, start_idx: int = 0, batch_size: int = 10):
        """Process a batch of cars from the DataFrame"""
        
        print(f"\n🚀 Starting bulk upload from index {start_idx}")
        print(f"📊 Processing {len(cars_df)} total cars in batches of {batch_size}")
        
        self.send_status({
            "status": "started",
            "batch_size": batch_size,
            "start_index": start_idx,
            "total_cars": len(cars_df)
        })
        
        # Get database session
        from database import engine, SessionLocal
        db = SessionLocal()
        
        try:
            processed = 0
            skipped = 0
            errors = 0
            
            for idx, row in cars_df.iterrows():
                if idx < start_idx:
                    continue
                    
                try:
                    make = str(row['NameName']).strip()
                    model = str(row['ModelName']).strip()
                    year = int(row['YearofManufactor'])
                    
                    print(f"\n📱 Processing {idx + 1}/{len(cars_df)}: {make} {model} {year}")
                    
                    self.send_status({
                        "status": "processing",
                        "current_index": idx + 1,
                        "total_cars": len(cars_df),
                        "make": make,
                        "model": model,
                        "year": year
                    })
                    
                    # Check if already exists
                    existing = db.query(CarReference).filter(
                        CarReference.model == model,
                        CarReference.year == year
                    ).first()
                    
                    if existing:
                        print(f"⏭️ Skipping - already exists: {make} {model} {year}")
                        skipped += 1
                        continue
                    
                    # Create directory for this car
                    car_dir = f"uploads/reference/{make}_{model}_{year}"
                    os.makedirs(car_dir, exist_ok=True)
                    
                    # Download images
                    image_urls = {
                        'front': str(row['Fcarimage']),
                        'back': str(row['Bcarimage']),
                        'left': str(row['Lcarimage']),
                        'right': str(row['Rcarimage'])
                    }
                    
                    downloaded_paths = {}
                    download_success = True
                    
                    for view, url in image_urls.items():
                        if pd.isna(url) or url == 'nan':
                            print(f"⚠️ Missing {view} URL")
                            download_success = False
                            break
                            
                        file_extension = '.jpg'  # IMAGIN.studio typically serves JPGs
                        filename = f"{view}_{uuid.uuid4().hex}{file_extension}"
                        file_path = os.path.join(car_dir, filename)
                        
                        if self.download_image(url, file_path):
                            downloaded_paths[view] = file_path
                        else:
                            print(f"❌ Failed to download {view} image")
                            download_success = False
                            break
                    
                    if not download_success:
                        # Clean up partial downloads
                        if os.path.exists(car_dir):
                            shutil.rmtree(car_dir)
                        errors += 1
                        continue
                    
                    # Extract features
                    print(f"🧠 Extracting features for {make} {model} {year}...")
                    
                    # Create visualization directory
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    vis_dir = f"static/visualizations/reference_{make}_{model}_{year}_{timestamp}"
                    os.makedirs(vis_dir, exist_ok=True)
                    
                    extracted_features = {}
                    processed_paths = {}
                    detection_paths = {}
                    
                    for side, path in downloaded_paths.items():
                        print(f"🔍 Processing {side} image...")
                        result = self.verifier.process_image(
                            path, 
                            save_visualizations=True, 
                            vis_dir=vis_dir, 
                            skip_preprocessing=False, 
                            is_reference=True
                        )
                        
                        if result['features'] is not None:
                            extracted_features[side] = serialize_features(result['features'])
                            processed_paths[side] = result.get('preprocessed_visualization')
                            detection_paths[side] = result.get('detection_visualization')
                            print(f"✅ {side} features extracted successfully")
                        else:
                            print(f"⚠️ Failed to extract features for {side}")
                            extracted_features[side] = None
                            processed_paths[side] = None
                            detection_paths[side] = None
                    
                    # Save to database
                    car_ref = CarReference(
                        model=model,
                        year=year,
                        description=f"Auto-imported from IMAGIN.studio - {make} {model}",
                        front_image_path=downloaded_paths.get("front"),
                        back_image_path=downloaded_paths.get("back"),
                        left_image_path=downloaded_paths.get("left"),
                        right_image_path=downloaded_paths.get("right"),
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
                    
                    processed += 1
                    print(f"✅ Successfully added: {make} {model} {year} (ID: {car_ref.id})")
                    
                    # Progress update
                    if processed % 5 == 0:
                        print(f"\n📊 Progress Update:")
                        print(f"   ✅ Processed: {processed}")
                        print(f"   ⏭️ Skipped: {skipped}")
                        print(f"   ❌ Errors: {errors}")
                        
                        self.send_status({
                            "status": "progress",
                            "processed": processed,
                            "skipped": skipped,
                            "errors": errors
                        })
                        
                    # Rate limiting - be nice to IMAGIN.studio
                    time.sleep(1)
                    
                    # Stop after batch size
                    if processed >= batch_size:
                        print(f"\n🎯 Batch completed! Processed {processed} cars.")
                        break
                        
                except Exception as e:
                    print(f"❌ Error processing car {idx}: {str(e)}")
                    errors += 1
                    
                    self.send_status({
                        "status": "error",
                        "error": str(e),
                        "car_index": idx
                    })
                    
                    continue
            
            print(f"\n🏁 Batch Summary:")
            print(f"   ✅ Successfully processed: {processed}")
            print(f"   ⏭️ Skipped (already exist): {skipped}")
            print(f"   ❌ Errors: {errors}")
            
            self.send_status({
                "status": "completed",
                "total_processed": processed,
                "skipped": skipped,
                "failed_cars": errors
            })
            
        finally:
            db.close()
    
    def run_bulk_upload(self, excel_file: str = "odysseydatasetcarimages.xlsx", 
                       start_idx: int = 0, batch_size: int = 20,
                       status_callback = None):
        """Main function to run bulk upload"""
        self.status_callback = status_callback
        
        try:
            print("📚 Reading Excel file...")
            df = pd.read_excel(excel_file)
            
            print(f"📊 Found {len(df)} cars in dataset")
            print(f"🎯 Starting from index {start_idx} with batch size {batch_size}")
            
            # Filter for popular brands first (optional)
            popular_brands = ['Toyota', 'Lexus', 'BMW', 'Mercedes', 'Audi', 'Honda', 'Nissan', 'Ford']
            
            print("🔍 Available brands in dataset:")
            brand_counts = df['NameName'].value_counts()
            print(brand_counts.head(10))
            
            # Option to process popular brands first
            df_filtered = df[df['NameName'].isin(popular_brands)].copy()
            if len(df_filtered) > 0:
                print(f"\n🎯 Found {len(df_filtered)} cars from popular brands")
                print("Processing popular brands first...")
                self.process_car_batch(df_filtered, start_idx, batch_size)
            else:
                print(f"\n🎯 Processing all brands...")
                self.process_car_batch(df, start_idx, batch_size)
                
        except Exception as e:
            print(f"❌ Error in bulk upload: {str(e)}")
            self.send_status({
                "status": "error",
                "error": str(e)
            })

def main():
    """Main function"""
    print("🚀 IMAGIN.studio Bulk Upload Tool")
    print("=" * 50)
    
    uploader = BulkImageUploader()
    
    # Start with a small batch for testing
    uploader.run_bulk_upload(
        excel_file="odysseydatasetcarimages.xlsx",
        start_idx=0,
        batch_size=10  # Start small for testing
    )

if __name__ == "__main__":
    main() 