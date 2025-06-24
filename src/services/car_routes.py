from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from bson.objectid import ObjectId
import os
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

from src.models.car import Car
from src.config.db import car_profiles_collection, users_collection

# Create blueprint
car_bp = Blueprint('car', __name__, url_prefix='/car')

# Define common car parts for insurance selection
CAR_PARTS = [
    'Front Bumper', 'Rear Bumper', 'Hood', 'Trunk/Boot', 'Front Left Door', 
    'Front Right Door', 'Rear Left Door', 'Rear Right Door', 'Left Fender', 
    'Right Fender', 'Roof', 'Grille', 'Headlights', 'Taillights', 'Mirrors',
    'Windows', 'Windshield', 'Engine', 'Transmission', 'Suspension', 'Brakes',
    'Exhaust System', 'Radiator', 'Battery', 'Fuel System', 'Electrical System'
]

@car_bp.route('/register', methods=['GET', 'POST'])
@login_required
def register_car():
    """Register a new car"""
    if request.method == 'POST':
        # Get car information from form
        car_info = {
            'make': request.form.get('make'),
            'model': request.form.get('model'),
            'year': int(request.form.get('year')),
            'color': request.form.get('color'),
            'license_plate': request.form.get('license_plate'),
            'insured_parts': request.form.getlist('insured_parts')
        }
        
        # Validate car information
        if not all([car_info['make'], car_info['model'], car_info['year'], 
                    car_info['color'], car_info['license_plate']]):
            flash('All car information is required.')
            return render_template('register_car.html', car_parts=CAR_PARTS)
        
        # Check if license plate is already in use
        existing_car = Car.get_by_license_plate(car_info['license_plate'])
        if existing_car:
            flash(f"License plate '{car_info['license_plate']}' is already registered to another car.")
            return render_template('register_car.html', car_parts=CAR_PARTS, car_info=car_info)
        
        # Process car images
        images = []
        image_types = ['front_image', 'side_image', 'rear_image']
        
        # Create upload directory for this user
        upload_dir = os.path.join('users_data', str(current_user.id), 'cars')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Process each image
        for image_type in image_types:
            if image_type in request.files and request.files[image_type].filename:
                # Get the file and generate a unique filename
                file = request.files[image_type]
                filename = secure_filename(f"{image_type}_{uuid.uuid4()}_{file.filename}")
                file_path = os.path.join(upload_dir, filename)
                
                # Save the file
                file.save(file_path)
                images.append(file_path)
        
        # Register the car with images
        car = Car.register_car(current_user.id, car_info, images)
        
        if car:
            flash('Car registered successfully!')
            return redirect(url_for('auth.profile'))
        else:
            flash('Failed to register car. Please try again.')
    
    # Display the registration form
    return render_template('register_car.html', car_parts=CAR_PARTS)

@car_bp.route('/view/<car_id>')
@login_required
def view_car(car_id):
    """View details of a specific car"""
    car = Car.get_by_id(car_id)
    
    # Check if car exists and belongs to the current user
    if not car or car.owner_id != str(current_user.id):
        flash('Car not found or access denied.')
        return redirect(url_for('auth.profile'))
    
    return render_template('view_car.html', car=car)

@car_bp.route('/edit/<car_id>', methods=['GET', 'POST'])
@login_required
def edit_car(car_id):
    """Edit car details"""
    car = Car.get_by_id(car_id)
    
    # Check if car exists and belongs to the current user
    if not car or car.owner_id != str(current_user.id):
        flash('Car not found or access denied.')
        return redirect(url_for('auth.profile'))
    
    if request.method == 'POST':
        # Update car information
        updates = {
            'make': request.form.get('make'),
            'model': request.form.get('model'),
            'year': int(request.form.get('year')),
            'color': request.form.get('color'),
            'license_plate': request.form.get('license_plate'),
            'insured_parts': request.form.getlist('insured_parts')
        }
        
        # Validate fields
        if not all([updates['make'], updates['model'], updates['year'], 
                    updates['color'], updates['license_plate']]):
            flash('All car information is required.')
            return render_template('edit_car.html', car=car, car_parts=CAR_PARTS)
        
        # Check if license plate is already in use by a different car
        if updates['license_plate'] != car.license_plate:
            existing_car = Car.get_by_license_plate(updates['license_plate'])
            if existing_car and existing_car.id != car_id:
                flash(f"License plate '{updates['license_plate']}' is already registered to another car.")
                return render_template('edit_car.html', car=car, car_parts=CAR_PARTS)
        
        # Update car in database
        car_profiles_collection.update_one(
            {'_id': ObjectId(car_id)},
            {'$set': updates}
        )
        
        # Process new images if any
        upload_dir = os.path.join('users_data', str(current_user.id), 'cars')
        os.makedirs(upload_dir, exist_ok=True)
        
        image_types = ['front_image', 'side_image', 'rear_image']
        for image_type in image_types:
            if image_type in request.files and request.files[image_type].filename:
                file = request.files[image_type]
                filename = secure_filename(f"{image_type}_{uuid.uuid4()}_{file.filename}")
                file_path = os.path.join(upload_dir, filename)
                file.save(file_path)
                
                # Add new image to car
                car.add_image(file_path)
        
        flash('Car information updated successfully!')
        return redirect(url_for('car.view_car', car_id=car_id))
    
    return render_template('edit_car.html', car=car, car_parts=CAR_PARTS)

# Note: Blueprint is registered directly in src/app_factory.py 