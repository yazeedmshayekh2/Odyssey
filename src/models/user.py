from flask_login import UserMixin
from bson.objectid import ObjectId

from src.config.db import users_collection

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data.get('_id'))
        self.username = user_data.get('username')
        self.email = user_data.get('email')
        self.first_name = user_data.get('first_name')
        self.last_name = user_data.get('last_name')
        self.registration_date = user_data.get('registration_date')
        self.cars = user_data.get('cars', [])
        self.license_image_path = user_data.get('license_image_path')
        self.car_image_path = user_data.get('car_image_path')
        self.documents_uploaded = user_data.get('documents_uploaded', False)
        
    @staticmethod
    def get_by_id(user_id):
        user_data = users_collection.find_one({'_id': ObjectId(user_id)})
        if not user_data:
            return None
        return User(user_data)
    
    @staticmethod
    def get_by_email(email):
        user_data = users_collection.find_one({'email': email})
        if not user_data:
            return None
        return User(user_data)
    
    @staticmethod
    def get_by_username(username):
        user_data = users_collection.find_one({'username': username})
        if not user_data:
            return None
        return User(user_data) 