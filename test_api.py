#!/usr/bin/env python3
"""
Test script for the updated car verification API
"""
import requests
import os
from pprint import pprint

def test_api():
    """Test the car verification API with sample images"""
    API_URL = "http://localhost:8000"
    
    print("🧪 Testing Car Verification API")
    print("=" * 50)
    
    # First check if API is running
    try:
        health = requests.get(f"{API_URL}/health")
        health.raise_for_status()
        print("✅ API Health Check:", health.json())
    except Exception as e:
        print(f"❌ API not available: {e}")
        return
    
    # Test data
    MODEL = "test_car"
    YEAR = 2024
    
    # Find some test images
    test_dirs = [
        "600-800/2024_05_26_19_00_11",  # Using the training data directory
        "uploads",
        "static"
    ]
    
    test_images = []
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(image_files) >= 4:  # We need at least 4 images
                test_images = [os.path.join(dir_path, f) for f in image_files[:4]]
                print(f"📸 Found test images in {dir_path}")
                break
    
    if not test_images:
        print("❌ Could not find enough test images")
        return
    
    print("\n1️⃣ Step 1: Upload Reference Images")
    print("-" * 50)
    
    # Upload reference images
    try:
        files = {
            'front_image': ('front.jpg', open(test_images[0], 'rb')),
            'back_image': ('back.jpg', open(test_images[1], 'rb')),
            'left_image': ('left.jpg', open(test_images[2], 'rb')),
            'right_image': ('right.jpg', open(test_images[3], 'rb')),
            'model': (None, MODEL),
            'year': (None, str(YEAR))
        }
        
        response = requests.post(f"{API_URL}/upload-reference", files=files)
        response.raise_for_status()
        print("✅ Reference images uploaded successfully")
        print("📊 Response:", response.json())
        
    except Exception as e:
        print(f"❌ Error uploading reference images: {e}")
        return
    finally:
        # Close all opened files
        for file in files.values():
            if hasattr(file[1], 'close'):
                file[1].close()
    
    print("\n2️⃣ Step 2: Test Verification")
    print("-" * 50)
    
    # Now test verification with the same images (should get high similarity)
    try:
        files = {
            'upload_front': ('front.jpg', open(test_images[0], 'rb')),
            'upload_back': ('back.jpg', open(test_images[1], 'rb')),
            'upload_left': ('left.jpg', open(test_images[2], 'rb')),
            'upload_right': ('right.jpg', open(test_images[3], 'rb'))
        }
        
        response = requests.post(f"{API_URL}/verify/{MODEL}/{YEAR}", files=files)
        response.raise_for_status()
        
        results = response.json()
        
        print("\n📊 Verification Results:")
        print("-" * 30)
        
        # Print individual side results
        for side in ["front", "back", "left", "right"]:
            result = results[f"{side}_result"]
            print(f"\n{side.upper()} View:")
            print(f"Match: {'✅' if result['is_match'] else '❌'}")
            print(f"Confidence: {result['confidence']}")
            print(f"Similarity Score: {result['similarity_score']:.2%}")
            if result.get('error'):
                print(f"Error: {result['error']}")
        
        # Print overall results
        print("\nOVERALL RESULTS:")
        print("-" * 30)
        print(f"Overall Match: {'✅' if results['overall_match'] else '❌'}")
        print(f"Average Similarity: {results['average_similarity']:.2%}")
        print(f"Overall Confidence: {results['overall_confidence']}")
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        if hasattr(e, 'response'):
            print("Response:", e.response.text)
    finally:
        # Close all opened files
        for file in files.values():
            if hasattr(file[1], 'close'):
                file[1].close()

if __name__ == "__main__":
    test_api() 