from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB configuration
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
DB_NAME = os.getenv('DB_NAME', 'car_insurance')

# Initialize MongoDB client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Define collections
users_collection = db.users
car_profiles_collection = db.car_profiles
damage_reports_collection = db.damage_reports 

def get_db():
    """Get database instance"""
    return db 