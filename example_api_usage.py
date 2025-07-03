import requests
from pathlib import Path
from typing import Dict, List
from pprint import pprint

def verify_car_images(
    api_url: str,
    ref_images: Dict[str, str],
    upload_images: Dict[str, str]
) -> Dict:
    """
    Verify car images using the verification API.
    
    Parameters:
    - api_url: Base URL of the verification API
    - ref_images: Dict with paths to reference images (front, back, left, right)
    - upload_images: Dict with paths to images to verify (front, back, left, right)
    
    Returns:
    - API response with verification results
    """
    # Prepare files for upload
    files = {
        # Reference images
        'ref_front': ('ref_front.jpg', open(ref_images['front'], 'rb')),
        'ref_back': ('ref_back.jpg', open(ref_images['back'], 'rb')),
        'ref_left': ('ref_left.jpg', open(ref_images['left'], 'rb')),
        'ref_right': ('ref_right.jpg', open(ref_images['right'], 'rb')),
        # Upload images
        'upload_front': ('upload_front.jpg', open(upload_images['front'], 'rb')),
        'upload_back': ('upload_back.jpg', open(upload_images['back'], 'rb')),
        'upload_left': ('upload_left.jpg', open(upload_images['left'], 'rb')),
        'upload_right': ('upload_right.jpg', open(upload_images['right'], 'rb')),
    }
    
    try:
        # Make API request
        response = requests.post(f"{api_url}/verify/", files=files)
        response.raise_for_status()
        
        # Parse and return results
        return response.json()
        
    finally:
        # Close all opened files
        for file in files.values():
            file[1].close()

def main():
    # API endpoint
    API_URL = "http://localhost:8000"
    
    # Example image paths
    ref_images = {
        "front": "reference/car1_front.jpg",
        "back": "reference/car1_back.jpg",
        "left": "reference/car1_left.jpg",
        "right": "reference/car1_right.jpg"
    }
    
    upload_images = {
        "front": "uploads/car2_front.jpg",
        "back": "uploads/car2_back.jpg",
        "left": "uploads/car2_left.jpg",
        "right": "uploads/car2_right.jpg"
    }
    
    # Check API health first
    try:
        health = requests.get(f"{API_URL}/health")
        health.raise_for_status()
        print("API Health Check:", health.json())
    except Exception as e:
        print(f"API not available: {e}")
        return
    
    # Verify images
    try:
        results = verify_car_images(API_URL, ref_images, upload_images)
        
        # Print results
        print("\nVerification Results:")
        print("-" * 50)
        
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
        print("-" * 50)
        print(f"Overall Match: {'✅' if results['overall_match'] else '❌'}")
        print(f"Average Similarity: {results['average_similarity']:.2%}")
        print(f"Overall Confidence: {results['overall_confidence']}")
        
    except Exception as e:
        print(f"Error during verification: {e}")

if __name__ == "__main__":
    main() 