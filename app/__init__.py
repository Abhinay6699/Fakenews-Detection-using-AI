from flask import Flask

def create_app():
    """
    Factory function to create the Flask application instance.
    """
    app = Flask(__name__)
    
    # Import and register blueprints
    from app.routes import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    return app
