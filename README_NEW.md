# Car Insurance System

A comprehensive car insurance system with damage detection capabilities. This application allows users to register their cars, upload documents, and report car damage. The system uses computer vision to detect and assess car damage.

## Features

- User registration and authentication
- Upload and management of car documents
- Car registration with multiple views
- Damage detection and reporting
- Integration with OCR for license plate recognition
- Insurance coverage management

## Project Structure

```
├── app.py                  # Main application entry point
├── src/                    # Source code directory
│   ├── auth/               # Authentication module
│   ├── models/             # Database models
│   ├── services/           # Business logic services
│   ├── utils/              # Utility functions
│   └── config/             # Configuration files
├── templates/              # HTML templates
├── static/                 # Static files (CSS, JS, images)
├── users_data/             # User-uploaded files (organized by user ID)
├── output/                 # Output files from damage detection
└── model_output/           # Machine learning model outputs
```

## Requirements

- Python 3.8+
- Flask
- MongoDB
- OpenCV
- PyTorch
- Detectron2
- Other dependencies in requirements.txt

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/car-insurance-system.git
cd car-insurance-system
```

### 2. Create and activate a virtual environment

```bash
python -m venv car_damage_env
source car_damage_env/bin/activate  # On Windows: car_damage_env\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### Install Detectron2

```bash
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
# Or for the latest version:
pip install --no-isolation 'git+https://github.com/facebookresearch/detectron2.git'
```

### 4. Set up MongoDB

See [SETUP_MONGODB.md](SETUP_MONGODB.md) for detailed MongoDB setup instructions.

### 5. Configure environment variables

Create a `.env` file in the root directory:

```
# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017/
DB_NAME=car_insurance

# Flask Configuration
SECRET_KEY=your_secret_key_here
DEBUG=True
PORT=5000

# Upload Configuration
MAX_CONTENT_LENGTH=16777216  # 16MB in bytes
```

### 6. Create necessary directories

```bash
mkdir -p output/car_damage_detection/uploads_test output/car_damage_detection/results_test users_data static
```

### 7. Run the application

```bash
python app.py
```

The application will be available at http://localhost:5000

## Usage

1. Register a new account
2. Upload your car license and photos
3. Register your car details
4. Report any damage to your car
5. View assessment reports

## OCR Integration

The system uses OCR to extract information from car licenses. The extracted data is used to pre-fill car registration forms.

## Damage Detection

The application uses a pre-trained damage detection model based on Detectron2 and Mask R-CNN. It can detect various types of car damage:

- Dents
- Scratches
- Cracks
- Glass shattering
- Broken lamps
- Flat tires

## License

[MIT License](LICENSE) 