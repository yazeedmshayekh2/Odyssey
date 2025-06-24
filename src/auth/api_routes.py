from flask import Blueprint, request, jsonify, current_app
from flask_login import login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from bson.objectid import ObjectId
import os
from datetime import datetime, timedelta
import uuid
import jwt

from src.models.user import User
from src.config.db import users_collection, car_profiles_collection
from src.auth.utils import bcrypt

# Create a blueprint for authentication API
auth_api_bp = Blueprint('auth_api', __name__, url_prefix='/api/auth')

# JWT Configuration
JWT_EXPIRATION = 24  # hours

def get_jwt_secret():
    """Get JWT secret from app config"""
    return current_app.config.get('JWT_SECRET', 'default_secret_key')

def generate_token(user_id):
    """Generate JWT token for user"""
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION)
    }
    return jwt.encode(payload, get_jwt_secret(), algorithm='HS256')

def authenticate_request():
    """Authenticate a request using JWT"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return None
    
    token = auth_header.split(' ')[1]
    
    try:
        payload = jwt.decode(token, get_jwt_secret(), algorithms=['HS256'])
        return payload.get('user_id')
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

@auth_api_bp.route('/register', methods=['POST'])
def register():
    """API endpoint for user registration"""
    data = request.get_json()
    
    if not data:
        return jsonify({'success': False, 'message': 'No data provided'}), 400
        
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    confirm_password = data.get('confirmPassword')
    first_name = data.get('firstName')
    last_name = data.get('lastName')
    
    # Validate data
    if not all([username, email, password, confirm_password, first_name, last_name]):
        return jsonify({'success': False, 'message': 'All fields are required'}), 400
    
    if password != confirm_password:
        return jsonify({'success': False, 'message': 'Passwords do not match'}), 400
    
    # Check if user already exists
    if User.get_by_email(email):
        return jsonify({'success': False, 'message': 'Email already registered'}), 400
    
    if User.get_by_username(username):
        return jsonify({'success': False, 'message': 'Username already taken'}), 400
    
    # Create new user
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = {
        'username': username,
        'email': email,
        'password': hashed_password,
        'first_name': first_name,
        'last_name': last_name,
        'registration_date': datetime.utcnow(),
        'cars': []
    }
    
    # Insert into database
    user_id = users_collection.insert_one(new_user).inserted_id
    
    # Create and return token
    token = generate_token(str(user_id))
    
    return jsonify({
        'success': True,
        'message': 'Registration successful',
        'token': token,
        'user': {
            'id': str(user_id),
            'username': username,
            'email': email,
            'firstName': first_name,
            'lastName': last_name
        }
    })

@auth_api_bp.route('/login', methods=['POST'])
def login():
    """API endpoint for user login"""
    data = request.get_json()
    
    if not data:
        return jsonify({'success': False, 'message': 'No data provided'}), 400
    
    email = data.get('email')
    password = data.get('password')
    
    # Find user by email
    user_data = users_collection.find_one({'email': email})
    
    if not user_data:
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    
    # Check password
    if not bcrypt.check_password_hash(user_data['password'], password):
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    
    # Generate token
    token = generate_token(str(user_data['_id']))
    
    return jsonify({
        'success': True,
        'message': 'Login successful',
        'token': token,
        'user': {
            'id': str(user_data['_id']),
            'username': user_data['username'],
            'email': user_data['email'],
            'firstName': user_data['first_name'],
            'lastName': user_data['last_name']
        }
    })

@auth_api_bp.route('/me', methods=['GET'])
def me():
    """Get current user's profile information"""
    user_id = authenticate_request()
    if not user_id:
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    
    # Get user data
    user_data = users_collection.find_one({'_id': ObjectId(user_id)})
    if not user_data:
        return jsonify({'success': False, 'message': 'User not found'}), 404
    
    # Build response
    user = {
        'id': str(user_data['_id']),
        'username': user_data['username'],
        'email': user_data['email'],
        'firstName': user_data['first_name'],
        'lastName': user_data['last_name'],
        'registrationDate': user_data['registration_date'],
        'documentsUploaded': user_data.get('documents_uploaded', False),
        'licenseImagePath': user_data.get('license_image_path'),
        'carImagePath': user_data.get('car_image_path')
    }
    
    return jsonify({
        'success': True,
        'user': user
    })

@auth_api_bp.route('/upload-documents', methods=['POST'])
def upload_documents():
    """Upload user documents (license and car image)"""
    user_id = authenticate_request()
    if not user_id:
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    
    # Check if license file is uploaded
    if 'licenseImage' not in request.files:
        return jsonify({'success': False, 'message': 'No license file part'}), 400
        
    # Check if car image is uploaded
    if 'carImage' not in request.files:
        return jsonify({'success': False, 'message': 'No car image file part'}), 400
        
    license_file = request.files['licenseImage']
    car_file = request.files['carImage']
    
    # Check if files are selected
    if license_file.filename == '' or car_file.filename == '':
        return jsonify({'success': False, 'message': 'No files selected'}), 400
    
    # Create uploads directory if it doesn't exist
    UPLOADS_DIR = os.path.join('users_data', user_id, 'documents')
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    
    # Save the license image
    license_filename = secure_filename(f"license_{uuid.uuid4()}_{license_file.filename}")
    license_path = os.path.join(UPLOADS_DIR, license_filename)
    license_file.save(license_path)
    
    # Save the car image
    car_filename = secure_filename(f"car_{uuid.uuid4()}_{car_file.filename}")
    car_path = os.path.join(UPLOADS_DIR, car_filename)
    car_file.save(car_path)
    
    # Update user document with the file paths
    users_collection.update_one(
        {'_id': ObjectId(user_id)},
        {'$set': {
            'license_image_path': license_path,
            'car_image_path': car_path,
            'documents_uploaded': True,
            'documents_upload_date': datetime.utcnow()
        }}
    )
    
    return jsonify({
        'success': True,
        'message': 'Documents uploaded successfully',
        'licenseImagePath': license_path,
        'carImagePath': car_path
    })

# Note: Blueprint is registered directly in src/auth/__init__.py 