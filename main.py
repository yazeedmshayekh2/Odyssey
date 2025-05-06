from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import numpy as np
from PIL import Image
import io
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from pathlib import Path
import uvicorn

# Create necessary directories
Path("templates").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)
Path("static/css").mkdir(exist_ok=True)

# Create a basic CSS file
css_content = """
.upload-area {
    border: 2px dashed #cbd5e0;
    transition: all 0.3s ease;
}
.upload-area:hover {
    border-color: #4299e1;
}
.loading {
    opacity: 0.7;
    pointer-events: none;
}
"""

# Write CSS file
with open("static/css/style.css", "w") as f:
    f.write(css_content)

# Create FastAPI app and mount static files
app = FastAPI(
    title="Car Damage Detection",
    description="API for detecting car damage using deep learning",
    version="1.0.0"
)

# Setup templates and static files AFTER creating directories
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model and processor globally
processor = AutoImageProcessor.from_pretrained("beingamit99/car_damage_detection")
model = AutoModelForImageClassification.from_pretrained("beingamit99/car_damage_detection")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {"request": request}
    )

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Make prediction
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probabilities = probabilities.detach().cpu().numpy()[0]
        
        # Get prediction details
        predicted_class_id = np.argmax(probabilities)
        predicted_proba = float(probabilities[predicted_class_id])
        label_map = model.config.id2label
        predicted_class_name = label_map[predicted_class_id]
        
        return {
            "class_name": predicted_class_name,
            "probability": f"{predicted_proba:.1%}",
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error processing image: {str(e)}")

# Create templates directory and index.html
Path("templates").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

# Create index.html
index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Car Damage Detection</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        .upload-area {
            border: 2px dashed #cbd5e0;
            transition: all 0.3s ease;
        }
        .upload-area:hover {
            border-color: #4299e1;
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-4xl font-bold text-center text-gray-800 mb-8">
            Car Damage Detection
        </h1>
        
        <div class="max-w-xl mx-auto bg-white rounded-lg shadow-lg p-6">
            <div id="upload-area" 
                 class="upload-area rounded-lg p-8 text-center cursor-pointer">
                <div id="preview" class="mb-4">
                    <img id="preview-img" class="hidden mx-auto max-h-64">
                </div>
                <p class="text-gray-600">
                    Drag and drop an image or click to select
                </p>
                <input type="file" id="file-input" class="hidden" accept="image/*">
            </div>
            
            <div id="result" class="mt-6 hidden">
                <h3 class="text-xl font-semibold mb-2">Results:</h3>
                <div class="bg-gray-50 rounded p-4">
                    <p class="mb-2">
                        <span class="font-medium">Damage Type:</span>
                        <span id="damage-type" class="ml-2"></span>
                    </p>
                    <p>
                        <span class="font-medium">Confidence:</span>
                        <span id="confidence" class="ml-2"></span>
                    </p>
                </div>
            </div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        const preview = document.getElementById('preview-img');
        const result = document.getElementById('result');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('border-blue-500');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('border-blue-500');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('border-blue-500');
            handleFile(e.dataTransfer.files[0]);
        });
        
        fileInput.addEventListener('change', (e) => {
            handleFile(e.target.files[0]);
        });
        
        function handleFile(file) {
            if (file && file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    preview.src = e.target.result;
                    preview.classList.remove('hidden');
                    uploadImage(file);
                };
                reader.readAsDataURL(file);
            }
        }
        
        async function uploadImage(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                if (data.success) {
                    document.getElementById('damage-type').textContent = data.class_name;
                    document.getElementById('confidence').textContent = data.probability;
                    result.classList.remove('hidden');
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error processing image');
            }
        }
    </script>
</body>
</html>
"""

# Write index.html
with open("templates/index.html", "w") as f:
    f.write(index_html)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)
