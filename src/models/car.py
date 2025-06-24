from bson.objectid import ObjectId
from datetime import datetime

from src.config.db import car_profiles_collection, users_collection

class Car:
    def __init__(self, car_data):
        self.id = str(car_data.get('_id'))
        self.make = car_data.get('make')
        self.model = car_data.get('model')
        self.year = car_data.get('year')
        self.color = car_data.get('color')
        self.license_plate = car_data.get('license_plate')
        self.owner_id = car_data.get('owner_id')
        self.registration_date = car_data.get('registration_date')
        self.images = car_data.get('images', [])
        self.insured_parts = car_data.get('insured_parts', [])
        self.damage_reports = car_data.get('damage_reports', [])
        # New fields
        self.chassis_no = car_data.get('chassis_no', '')
        self.engine_no = car_data.get('engine_no', '')
        self.seats = car_data.get('seats', 0)
        self.weight = car_data.get('weight', 0)
        self.total_weight = car_data.get('total_weight', 0)
        self.country_origin = car_data.get('country_origin', '')
        self.engine_type = car_data.get('engine_type', '')
        self.insurance_expiry = car_data.get('insurance_expiry', '')
        self.insurance_policy = car_data.get('insurance_policy', '')
        
    @staticmethod
    def get_by_id(car_id):
        """Retrieve a car by its ID"""
        car_data = car_profiles_collection.find_one({'_id': ObjectId(car_id)})
        if not car_data:
            return None
        return Car(car_data)
    
    @staticmethod
    def get_by_owner(owner_id):
        """Get all cars owned by a specific user"""
        cars_data = car_profiles_collection.find({'owner_id': str(owner_id)})
        return [Car(car_data) for car_data in cars_data]
    
    @staticmethod
    def get_by_license_plate(license_plate):
        """Find a car by license plate"""
        car_data = car_profiles_collection.find_one({'license_plate': license_plate})
        if not car_data:
            return None
        return Car(car_data)
        
    @staticmethod
    def register_car(owner_id, car_info, image_paths=None):
        """Register a new car for a user"""
        # Create new car document
        new_car = {
            'make': car_info.get('make'),
            'model': car_info.get('model'),
            'year': car_info.get('year'),
            'color': car_info.get('color'),
            'license_plate': car_info.get('license_plate'),
            'owner_id': str(owner_id),
            'registration_date': datetime.utcnow(),
            'images': image_paths or [],
            'insured_parts': car_info.get('insured_parts', []),
            # New fields
            'chassis_no': car_info.get('chassis_no', ''),
            'engine_no': car_info.get('engine_no', ''),
            'seats': car_info.get('seats', 0),
            'weight': car_info.get('weight', 0),
            'total_weight': car_info.get('total_weight', 0),
            'country_origin': car_info.get('country_origin', ''),
            'engine_type': car_info.get('engine_type', ''),
            'insurance_expiry': car_info.get('insurance_expiry', ''),
            'insurance_policy': car_info.get('insurance_policy', '')
        }
        
        # Insert car into database
        car_id = car_profiles_collection.insert_one(new_car).inserted_id
        
        # Update user's cars list
        users_collection.update_one(
            {'_id': ObjectId(owner_id)},
            {'$push': {'cars': car_id}}
        )
        
        return Car.get_by_id(car_id)
        
    def add_image(self, image_path):
        """Add a new image to the car's image collection"""
        car_profiles_collection.update_one(
            {'_id': ObjectId(self.id)},
            {'$push': {'images': image_path}}
        )
        # Update the object's images list
        self.images.append(image_path)
        
    def update_insurance(self, insured_parts):
        """Update the car's insured parts"""
        car_profiles_collection.update_one(
            {'_id': ObjectId(self.id)},
            {'$set': {'insured_parts': insured_parts}}
        )
        self.insured_parts = insured_parts
        
    def add_damage_report(self, damage_report_id):
        """Add a damage report reference to the car"""
        car_profiles_collection.update_one(
            {'_id': ObjectId(self.id)},
            {'$push': {'damage_reports': damage_report_id}}
        )
        self.damage_reports.append(damage_report_id) 