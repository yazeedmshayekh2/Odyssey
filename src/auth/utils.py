from flask_login import LoginManager
from flask_bcrypt import Bcrypt

# Create instances
bcrypt = Bcrypt()

def init_login_manager(app):
    """Initialize and configure the login manager"""
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    
    # Initialize bcrypt with the application
    bcrypt.init_app(app)
    
    # User loader callback
    from src.models.user import User
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.get_by_id(user_id)
    
    return login_manager 