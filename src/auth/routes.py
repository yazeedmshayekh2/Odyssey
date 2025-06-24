from flask import render_template, redirect, url_for, flash, request, Blueprint
from flask_login import login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from bson.objectid import ObjectId
import os
from datetime import datetime
import uuid

from src.models.user import User
from src.config.db import users_collection, car_profiles_collection
from src.auth.utils import bcrypt

# Create a blueprint for user authentication
auth_bp = Blueprint('auth', __name__)

# Routes for authentication
@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get form data
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        
        # Validate data
        if not all([username, email, password, confirm_password, first_name, last_name]):
            flash('All fields are required')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match')
            return render_template('register.html')
        
        # Check if user already exists
        if User.get_by_email(email):
            flash('Email already registered')
            return render_template('register.html')
        
        if User.get_by_username(username):
            flash('Username already taken')
            return render_template('register.html')
        
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
        
        # Log in the newly registered user
        user = User.get_by_id(user_id)
        login_user(user)
        
        flash('Registration successful! Please complete your profile.')
        return redirect(url_for('auth.upload_documents'))
    
    return render_template('register.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Find user by email
        user_data = users_collection.find_one({'email': email})
        
        if user_data and bcrypt.check_password_hash(user_data['password'], password):
            user = User(user_data)
            login_user(user)
            
            # Redirect to the requested page or default to profile page
            next_page = request.args.get('next')
            return redirect(next_page or url_for('auth.profile'))
        else:
            flash('Login failed. Please check your email and password.')
    
    return render_template('login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.')
    return redirect(url_for('auth.login'))

# Upload user documents (license and car image)
@auth_bp.route('/upload-documents', methods=['GET', 'POST'])
@login_required
def upload_documents():
    if request.method == 'POST':
        # Check if license file is uploaded
        if 'license_image' not in request.files:
            flash('No license file part')
            return render_template('upload_documents.html')
            
        # Check if car image is uploaded
        if 'car_image' not in request.files:
            flash('No car image file part')
            return render_template('upload_documents.html')
            
        license_file = request.files['license_image']
        car_file = request.files['car_image']
        
        # Check if files are selected
        if license_file.filename == '' or car_file.filename == '':
            flash('No files selected')
            return render_template('upload_documents.html')
        
        # Create uploads directory if it doesn't exist
        UPLOADS_DIR = os.path.join('users_data', str(current_user.id), 'documents')
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
            {'_id': ObjectId(current_user.id)},
            {'$set': {
                'license_image_path': license_path,
                'car_image_path': car_path,
                'documents_uploaded': True,
                'documents_upload_date': datetime.utcnow()
            }}
        )
        
        flash('Documents uploaded successfully!')
        return redirect(url_for('auth.profile'))
    
    return render_template('upload_documents.html')

@auth_bp.route('/profile')
@login_required
def profile():
    # Get user information
    user_data = users_collection.find_one({'_id': ObjectId(current_user.id)})
    
    # Get car profiles
    user_cars = []
    for car_id in user_data.get('cars', []):
        car_profile = car_profiles_collection.find_one({'_id': car_id})
        if car_profile:
            user_cars.append(car_profile)
            
    return render_template('profile.html', user=user_data, cars=user_cars) 