# 🚗 Car Verification System - AI-Powered Vehicle Matching

An advanced AI-powered car image verification system that uses **InceptionV3** and **YOLOv11** to determine if uploaded car images match stored reference images. This system combines powerful deep learning architectures for reliable vehicle identification and verification.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🧠 AI Technology Stack

### Core AI Models
- **InceptionV3**: Deep convolutional neural network with multi-scale feature extraction
- **YOLOv11**: Latest YOLO model for accurate car detection and cropping
- **Global Average Pooling**: Effective feature aggregation from convolutional layers
- **Cosine Similarity**: Proven similarity metric for deep learning feature comparison

### Advanced Features
- **Deep Feature Extraction**: InceptionV3 captures 2048-dimensional high-level semantic features
- **Smart Car Detection**: Automatically detects and crops car regions using YOLOv11
- **Adaptive Preprocessing**: Enhanced preprocessing pipeline optimized for car verification
- **L2 Normalization**: Normalized feature vectors for stable similarity computation
- **GPU Acceleration**: Automatic CUDA utilization when available

## 🎯 Key Capabilities

### Proven Accuracy
- **85%+ Similarity Threshold**: Reliable matching with InceptionV3 features
- **ImageNet Pre-trained**: Leverages pre-trained weights from millions of images
- **Robust Feature Representation**: 2048-dimensional feature vectors
- **Multi-Confidence Levels**: High (90%+), Medium (80%+), Low (<80%)

### Intelligent Processing
- **Automatic Car Detection**: YOLOv11 identifies car regions before feature extraction
- **Smart Cropping**: Adaptive padding based on detected car dimensions
- **Enhanced Preprocessing**: Advanced image processing pipeline for better feature extraction
- **Robust Error Handling**: Graceful fallback to full image if car detection fails

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for acceleration)
- 4GB+ RAM recommended

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/OdysseyModelTraining.git
   cd OdysseyModelTraining
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize database:**
   ```bash
   python database.py
   ```

### Running the Application

#### Web Interface
```bash
python main.py
```
The web application will be available at `http://localhost:8000`

#### API Server
```bash
uvicorn car_verification_api:app --host 0.0.0.0 --port 8000
```
The API will be available at `http://localhost:8000`

### Docker Deployment

1. **Build the Docker image:**
   ```bash
   docker build -t car-verification-api .
   ```

2. **Run the container:**
   ```bash
   docker run -d -p 8000:8000 car-verification-api
   ```

## 🔧 Usage

### Web Interface

#### 1. Upload Reference Images
- Navigate to the "Upload Reference" tab
- Fill in car model and year
- Upload 4 images (front, back, left, right views)
- System processes images with InceptionV3 AI automatically

#### 2. Verify Car Images
- Go to "Verify Car" tab
- Enter car model and year to verify against
- Upload 4 verification images
- Get detailed AI analysis with confidence scores

#### 3. View Stored References
- Check "View References" tab
- Browse all stored car references
- View processed images and AI visualizations

### API Usage

#### Upload Reference Images from URLs
```python
import requests

# Upload reference images
data = {
    'model': 'Toyota Land Cruiser',
    'year': 2022,
    'description': 'Toyota Land Cruiser 2022',
    'front_url': 'https://example.com/front.jpg',
    'back_url': 'https://example.com/back.jpg',
    'left_url': 'https://example.com/left.jpg',
    'right_url': 'https://example.com/right.jpg'
}

response = requests.post('http://localhost:8000/upload-reference-urls', data=data)
print(response.json())
```

#### Verify Car Images
```python
import requests

# Verify car images
files = {
    'upload_front': open('test_front.jpg', 'rb'),
    'upload_back': open('test_back.jpg', 'rb'),
    'upload_left': open('test_left.jpg', 'rb'),
    'upload_right': open('test_right.jpg', 'rb')
}

response = requests.post(
    'http://localhost:8000/verify/Toyota Land Cruiser/2022', 
    files=files
)
print(response.json())
```

## 📊 API Documentation

### Endpoints

#### Upload Reference Images
- **POST** `/upload-reference-urls`
- Upload reference images from URLs for a car model

#### Verify Car Images  
- **POST** `/verify/{model}/{year}`
- Verify uploaded images against stored reference

#### Health Check
- **GET** `/health`
- Check API status

### Response Format

#### Verification Response
```json
{
    "front_result": {
        "side": "front",
        "is_match": true,
        "confidence": "high",
        "similarity_score": 0.92
    },
    "back_result": { "..." },
    "left_result": { "..." },
    "right_result": { "..." },
    "overall_match": true,
    "average_similarity": 0.89,
    "overall_confidence": "high"
}
```

## 🔬 Technical Architecture

### Image Processing Pipeline
1. **YOLOv11 Detection**: Locates car objects in uploaded images
2. **Smart Cropping**: Extracts car regions with adaptive padding
3. **Enhanced Preprocessing**: Advanced image processing pipeline
4. **InceptionV3 Feature Extraction**: Generates 2048-dimensional feature vectors
5. **L2 Normalization**: Prepares features for similarity comparison
6. **Cosine Similarity**: Computes similarity between feature vectors

### Database Schema
- **CarReference**: Stores car model data, image paths, and extracted features
- **VerificationAttempt**: Logs all verification attempts with results
- **Feature Storage**: Serialized numpy arrays for efficient storage and retrieval

### Performance Characteristics
- **Feature Extraction**: 2048-dimensional deep features
- **Processing Speed**: ~1-3 seconds per 4-image set (GPU)
- **Memory Usage**: ~1GB VRAM (GPU mode)
- **Accuracy**: 85%+ matching accuracy with proper reference images

## 🛠️ Configuration

### Environment Variables
Create a `.env` file for configuration:
```env
DATABASE_URL=sqlite:///./car_database.db
MODEL_PATH=./yolo11x.pt
SIMILARITY_THRESHOLD=0.85
CAR_CONFIDENCE_THRESHOLD=0.5
```

### Similarity Thresholds
```python
SIMILARITY_THRESHOLD = 0.85      # Main matching threshold
CAR_CONFIDENCE_THRESHOLD = 0.5   # YOLO detection confidence
```

### Supported Image Formats
- JPEG, JPG, PNG, BMP, TIFF, WebP
- Recommended: High-resolution images (1024x1024+)
- Multiple angles required for best results

## 📁 Project Structure

```
OdysseyModelTraining/
├── main.py                     # Web interface application
├── car_verification_api.py     # API server
├── car_verification.py         # Core AI verification logic
├── database.py                 # Database models and setup
├── models.py                   # Pydantic models
├── requirements.txt            # Python dependencies
├── requirements_api.txt        # API-specific dependencies
├── Dockerfile                  # Docker configuration
├── templates/                  # HTML templates
│   └── index.html             # Web interface
├── uploads/                   # Uploaded images
│   ├── reference/             # Reference car images
│   └── verification/          # Verification attempt images
└── static/                    # Static files and visualizations
```

## 🧪 Testing

### API Testing
```bash
# Run API tests
python test_api.py

# Example API usage
python example_api_usage.py
```

### Web Interface Testing
1. Start the web server: `python main.py`
2. Navigate to `http://localhost:8000`
3. Upload reference images for a car model
4. Test verification with different images

## 🔍 Troubleshooting

### Common Issues

**YOLO model not found:**
```bash
# Download YOLO model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11x.pt
```

**InceptionV3 model not loading:**
```bash
pip install torch torchvision --upgrade
```

**CUDA out of memory:**
- Use CPU mode by setting `device = torch.device('cpu')`
- Reduce batch size or image resolution

**Low similarity scores:**
- Ensure good image quality and lighting
- Use clear, unobstructed car views
- Check that car detection is working properly

## 📚 Dependencies

### Core AI Libraries
- `torch>=2.1.1` - PyTorch deep learning framework
- `torchvision>=0.16.1` - Computer vision utilities and models
- `ultralytics>=8.3.160` - YOLOv11 object detection

### Web Framework
- `fastapi>=0.104.1` - Modern Python web framework
- `uvicorn>=0.24.0` - ASGI server
- `python-multipart>=0.0.6` - File upload support

### Image Processing
- `opencv-python>=4.8.1` - Computer vision operations
- `pillow>=10.1.0` - Image manipulation
- `numpy>=1.26.2` - Numerical computations

### Database
- `sqlalchemy>=2.0.23` - SQL toolkit and ORM
- SQLite database (included with Python)

## 🚀 Deployment

### Local Development
```bash
python main.py  # Web interface on port 8000
```

### Production API
```bash
uvicorn car_verification_api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Production
```bash
docker run -d -p 8000:8000 -v $(pwd)/uploads:/app/uploads car-verification-api
```

### Cloud Deployment
The application is ready for deployment on:
- AWS EC2/ECS
- Google Cloud Run
- Azure Container Instances
- Heroku
- DigitalOcean App Platform

## 🔐 Security Considerations

- **Authentication**: Currently no authentication - add API keys for production
- **Rate Limiting**: Implement rate limiting for production use
- **HTTPS**: Use HTTPS in production environments
- **Input Validation**: Comprehensive input validation implemented
- **File Security**: Uploaded files are validated and stored securely

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/OdysseyModelTraining/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/OdysseyModelTraining/discussions)
- **Documentation**: [Wiki](https://github.com/your-username/OdysseyModelTraining/wiki)

## 🏆 Acknowledgments

- **InceptionV3**: Google's powerful CNN architecture
- **YOLOv11**: Ultralytics' latest object detection model
- **PyTorch**: Facebook's deep learning framework
- **FastAPI**: Tiangolo's modern web framework

---

**Powered by InceptionV3 + YOLOv11** 🚗🧠 

*Built with ❤️ for accurate car verification* 