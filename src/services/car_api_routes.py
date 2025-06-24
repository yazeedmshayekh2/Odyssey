from flask import Blueprint, request, jsonify, current_app
from bson.objectid import ObjectId
import os
from werkzeug.utils import secure_filename
import uuid
import jwt
from datetime import datetime

from src.models.car import Car
from src.config.db import car_profiles_collection, users_collection

# Create blueprint for car API
car_api_bp = Blueprint('car_api', __name__, url_prefix='/api/cars')

# Common car parts for insurance selection
CAR_PARTS = [
    'Front Bumper', 'Rear Bumper', 'Hood', 'Trunk/Boot', 'Front Left Door', 
    'Front Right Door', 'Rear Left Door', 'Rear Right Door', 'Left Fender', 
    'Right Fender', 'Roof', 'Grille', 'Headlights', 'Taillights', 'Mirrors',
    'Windows', 'Windshield', 'Engine', 'Transmission', 'Suspension', 'Brakes',
    'Exhaust System', 'Radiator', 'Battery', 'Fuel System', 'Electrical System'
]

def get_jwt_secret():
    """Get JWT secret from app config"""
    return current_app.config.get('JWT_SECRET', 'default_secret_key')

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

@car_api_bp.route('/parts', methods=['GET'])
def get_car_parts():
    """Get list of car parts for insurance selection"""
    return jsonify({
        'success': True,
        'parts': CAR_PARTS
    })

@car_api_bp.route('/', methods=['GET'])
def get_user_cars():
    """Get all cars for the authenticated user"""
    user_id = authenticate_request()
    if not user_id:
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    
    cars = Car.get_by_owner(user_id)
    
    # Convert cars to JSON-serializable format
    cars_data = []
    for car in cars:
        car_data = {
            'id': car.id,
            'make': car.make,
            'model': car.model,
            'year': car.year,
            'color': car.color,
            'licensePlate': car.license_plate,
            'registrationDate': car.registration_date,
            'images': car.images,
            'insuredParts': car.insured_parts,
            'damageReports': car.damage_reports
        }
        cars_data.append(car_data)
    
    return jsonify({
        'success': True,
        'cars': cars_data
    })

@car_api_bp.route('/<car_id>', methods=['GET'])
def get_car(car_id):
    """Get details of a specific car"""
    user_id = authenticate_request()
    if not user_id:
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    
    car = Car.get_by_id(car_id)
    
    # Check if car exists and belongs to the current user
    if not car or car.owner_id != user_id:
        return jsonify({'success': False, 'message': 'Car not found or access denied'}), 404
    
    # Convert car to JSON-serializable format
    car_data = {
        'id': car.id,
        'make': car.make,
        'model': car.model,
        'year': car.year,
        'color': car.color,
        'licensePlate': car.license_plate,
        'registrationDate': car.registration_date,
        'images': car.images,
        'insuredParts': car.insured_parts,
        'damageReports': car.damage_reports,
        'chassisNo': car.chassis_no,
        'engineNo': car.engine_no,
        'seats': car.seats,
        'weight': car.weight,
        'totalWeight': car.total_weight,
        'countryOrigin': car.country_origin,
        'engineType': car.engine_type,
        'insuranceExpiry': car.insurance_expiry,
        'insurancePolicy': car.insurance_policy
    }
    
    return jsonify({
        'success': True,
        'car': car_data
    })

@car_api_bp.route('/', methods=['POST'])
def register_car():
    """Register a new car"""
    print(f"DEBUG: Car registration request received")
    print(f"DEBUG: Request method: {request.method}")
    print(f"DEBUG: Request headers: {dict(request.headers)}")
    print(f"DEBUG: Request form data: {request.form.to_dict()}")
    print(f"DEBUG: Request files: {list(request.files.keys())}")
    
    user_id = authenticate_request()
    print(f"DEBUG: Authenticated user_id: {user_id}")
    if not user_id:
        print("DEBUG: Authentication failed - no user_id")
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    
    # Get car information from JSON data
    data = request.form.to_dict()
    print(f"DEBUG: Form data parsed: {data}")
    
    try:
        car_info = {
            'make': data.get('make'),
            'model': data.get('model'),
                'year': int(data.get('year', 0)) if data.get('year') else 0,
            'color': data.get('color'),
            'license_plate': data.get('licensePlate'),
            'chassis_no': data.get('chassisNo', ''),
            'engine_no': data.get('engineNo', ''),
            'seats': int(data.get('seats', 0)) if data.get('seats') else 0,
            'weight': int(data.get('weight', 0)) if data.get('weight') else 0,
            'total_weight': int(data.get('totalWeight', 0)) if data.get('totalWeight') else 0,
            'country_origin': data.get('countryOrigin', ''),
            'engine_type': data.get('engineType', ''),
            'insurance_expiry': data.get('insuranceExpiry', ''),
            'insurance_policy': data.get('insurancePolicy', ''),
            'insured_parts': request.form.getlist('insuredParts')
        }
    except ValueError as e:
        return jsonify({
            'success': False, 
            'message': f'Invalid numeric value in form data: {e}'
        }), 400
    

    
    # Validate car information
    required_fields = ['make', 'model', 'year', 'color', 'license_plate']
    missing_fields = []
    
    for field in required_fields:
        if not car_info[field]:
            missing_fields.append(field)
    
    if missing_fields:
        return jsonify({
            'success': False, 
            'message': f'Missing required fields: {", ".join(missing_fields)}'
        }), 400
    
    # Check if license plate is already in use
    existing_car = Car.get_by_license_plate(car_info['license_plate'])
    if existing_car:
        return jsonify({
            'success': False, 
            'message': f"License plate '{car_info['license_plate']}' is already registered to another car"
        }), 400
    
    # Process car image
    images = []
    
    # Create upload directory for this user
    upload_dir = os.path.join('users_data', user_id, 'cars')
    os.makedirs(upload_dir, exist_ok=True)
    
    # Process car image if available
    if 'carImage' in request.files and request.files['carImage'].filename:
        # Get the file and generate a unique filename
        file = request.files['carImage']
        filename = secure_filename(f"car_{uuid.uuid4()}_{file.filename}")
        file_path = os.path.join(upload_dir, filename)
        
        # Save the file
        file.save(file_path)
        images.append(file_path)
    
    # Register the car with images
    car = Car.register_car(user_id, car_info, images)
    
    if not car:
        return jsonify({'success': False, 'message': 'Failed to register car'}), 500
    
    # Convert car to JSON-serializable format
    car_data = {
        'id': car.id,
        'make': car.make,
        'model': car.model,
        'year': car.year,
        'color': car.color,
        'licensePlate': car.license_plate,
        'registrationDate': car.registration_date,
        'images': car.images,
        'insuredParts': car.insured_parts,
        'damageReports': car.damage_reports,
        'chassisNo': car.chassis_no,
        'engineNo': car.engine_no,
        'seats': car.seats,
        'weight': car.weight,
        'totalWeight': car.total_weight,
        'countryOrigin': car.country_origin,
        'engineType': car.engine_type,
        'insuranceExpiry': car.insurance_expiry,
        'insurancePolicy': car.insurance_policy
    }
    
    return jsonify({
        'success': True,
        'message': 'Car registered successfully',
        'car': car_data
    }), 201

@car_api_bp.route('/<car_id>', methods=['PUT'])
def update_car(car_id):
    """Update car details"""
    user_id = authenticate_request()
    if not user_id:
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    
    car = Car.get_by_id(car_id)
    
    # Check if car exists and belongs to the current user
    if not car or car.owner_id != user_id:
        return jsonify({'success': False, 'message': 'Car not found or access denied'}), 404
    
    # Get updated car information from form data
    data = request.form.to_dict()
    
    updates = {
        'make': data.get('make'),
        'model': data.get('model'),
        'year': int(data.get('year', 0)),
        'color': data.get('color'),
        'license_plate': data.get('licensePlate'),
        'chassis_no': data.get('chassisNo', ''),
        'engine_no': data.get('engineNo', ''),
        'seats': int(data.get('seats', 0)) if data.get('seats') else 0,
        'weight': int(data.get('weight', 0)) if data.get('weight') else 0,
        'total_weight': int(data.get('totalWeight', 0)) if data.get('totalWeight') else 0,
        'country_origin': data.get('countryOrigin', ''),
        'engine_type': data.get('engineType', ''),
        'insurance_expiry': data.get('insuranceExpiry', ''),
        'insurance_policy': data.get('insurancePolicy', ''),
        'insured_parts': request.form.getlist('insuredParts')
    }
    
    # Validate fields
    if not all([updates['make'], updates['model'], updates['year'], 
                updates['color'], updates['license_plate']]):
        return jsonify({'success': False, 'message': 'All car information is required'}), 400
    
    # Check if license plate is already in use by a different car
    if updates['license_plate'] != car.license_plate:
        existing_car = Car.get_by_license_plate(updates['license_plate'])
        if existing_car and existing_car.id != car_id:
            return jsonify({
                'success': False, 
                'message': f"License plate '{updates['license_plate']}' is already registered to another car"
            }), 400
    
    # Update car in database
    car_profiles_collection.update_one(
        {'_id': ObjectId(car_id)},
        {'$set': updates}
    )
    
    # Process new car image if any
    upload_dir = os.path.join('users_data', user_id, 'cars')
    os.makedirs(upload_dir, exist_ok=True)
    
    new_images = []
    if 'carImage' in request.files and request.files['carImage'].filename:
        file = request.files['carImage']
        filename = secure_filename(f"car_{uuid.uuid4()}_{file.filename}")
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)
        
        # Add new image to car
        car.add_image(file_path)
        new_images.append(file_path)
    
    # Get updated car data
    updated_car = Car.get_by_id(car_id)
    
    # Convert car to JSON-serializable format
    car_data = {
        'id': updated_car.id,
        'make': updated_car.make,
        'model': updated_car.model,
        'year': updated_car.year,
        'color': updated_car.color,
        'licensePlate': updated_car.license_plate,
        'registrationDate': updated_car.registration_date,
        'images': updated_car.images,
        'insuredParts': updated_car.insured_parts,
        'damageReports': updated_car.damage_reports,
        'chassisNo': updated_car.chassis_no,
        'engineNo': updated_car.engine_no,
        'seats': updated_car.seats,
        'weight': updated_car.weight,
        'totalWeight': updated_car.total_weight,
        'countryOrigin': updated_car.country_origin,
        'engineType': updated_car.engine_type,
        'insuranceExpiry': updated_car.insurance_expiry,
        'insurancePolicy': updated_car.insurance_policy,
        'newImages': new_images
    }
    
    return jsonify({
        'success': True,
        'message': 'Car information updated successfully',
        'car': car_data
    })

@car_api_bp.route('/<car_id>', methods=['DELETE'])
def delete_car(car_id):
    """Delete a car"""
    user_id = authenticate_request()
    if not user_id:
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    
    car = Car.get_by_id(car_id)
    
    # Check if car exists and belongs to the current user
    if not car or car.owner_id != user_id:
        return jsonify({'success': False, 'message': 'Car not found or access denied'}), 404
    
    # Remove car from user's cars list
    users_collection.update_one(
        {'_id': ObjectId(user_id)},
        {'$pull': {'cars': ObjectId(car_id)}}
    )
    
    # Delete car from database
    car_profiles_collection.delete_one({'_id': ObjectId(car_id)})
    
    return jsonify({
        'success': True,
        'message': 'Car deleted successfully'
    })

# Note: Blueprint is registered directly in src/app_factory.py 