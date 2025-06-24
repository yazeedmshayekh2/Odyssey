from flask import Flask, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_config_value(key, default, value_type=str):
    """Get configuration value with proper type conversion and error handling"""
    value = os.getenv(key, default)
    if isinstance(value, str):
        # Strip whitespace and remove inline comments
        value = value.strip().split('#')[0].strip()
    
    if value_type == int:
        try:
            return int(value)
        except (ValueError, TypeError):
            return default if isinstance(default, int) else int(default)
    
    return value

def create_app():
    """
    Create and configure the Flask application.
    
    Returns:
        Flask application instance
    """
    app = Flask(__name__, 
                template_folder='../templates',
                static_folder='../static')
    
    # Configure the app with robust environment variable handling
    app.config['SECRET_KEY'] = get_config_value('SECRET_KEY', 'dev_secret_key_change_in_production')
    app.config['MONGO_URI'] = get_config_value('MONGO_URI', 'mongodb://localhost:27017/')
    app.config['DB_NAME'] = get_config_value('DB_NAME', 'car_insurance')
    app.config['MAX_CONTENT_LENGTH'] = get_config_value('MAX_CONTENT_LENGTH', 16 * 1024 * 1024, int)
    app.config['JWT_SECRET'] = get_config_value('JWT_SECRET', 'odyssey_secret_key_change_in_production')
    
    # Configure CORS with proper settings
    frontend_url = get_config_value('FRONTEND_URL', 'http://localhost:3000')
    CORS(app, 
         resources={
             r"/api/*": {
                 "origins": [
                     "http://localhost:3000", 
                     "http://localhost:5000", 
                     "http://localhost:5001",
                     frontend_url
                 ],
                 "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                 "allow_headers": ["Content-Type", "Authorization"],
                 "supports_credentials": True
             },
             r"/detect_damage": {
                 "origins": [
                     "http://localhost:3000", 
                     "http://localhost:5000", 
                     "http://localhost:5001",
                     frontend_url
                 ],
                 "methods": ["POST", "OPTIONS"],
                 "allow_headers": ["Content-Type", "Authorization"],
                 "supports_credentials": True
             }
         })
    
    # Add health check route
    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({'status': 'ok', 'message': 'API is running'})
    
    # Register blueprints
    from src.auth import init_auth
    from src.services.car_routes import car_bp
    from src.services.car_api_routes import car_api_bp
    from src.services.damage_api_routes import damage_api_bp
    
    # Initialize authentication (registers auth blueprints)
    init_auth(app)
    
    # Register car routes directly
    app.register_blueprint(car_bp)
    app.register_blueprint(car_api_bp)
    app.register_blueprint(damage_api_bp)
    
    return app 