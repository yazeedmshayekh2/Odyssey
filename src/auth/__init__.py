"""
Authentication Module
-------------------
Handles user authentication, registration, and profile management.
"""

from src.auth.routes import auth_bp
from src.auth.api_routes import auth_api_bp

def init_auth(app):
    """Initialize the auth module with the Flask application"""
    from src.auth.utils import init_login_manager
    
    # Register the auth blueprint with the app (for web templates)
    app.register_blueprint(auth_bp)
    
    # Register the auth API blueprint (for REST API)
    app.register_blueprint(auth_api_bp)
    
    # Initialize login manager
    init_login_manager(app)
    
    return app 