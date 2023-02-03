from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

db = SQLAlchemy()


def create_app():
    app = Flask(__name__)

    app.config['SECRET_KEY'] = '9e535854a2b5194a97225b2d2c4ea5b89adda5ae0a1f2f8f4d42c688d86158cd'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite://///root/workspace/engineering-project/application/db.sqlite'

    db.init_app(app)

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    from .models import User

    @login_manager.user_loader
    def load_user(user_id):
        # since the user_id is just the primary key of our user table, use it in the query for the user
        return User.query.get(int(user_id))

    from .auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint)

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    # from .upload_file import upload_file as upload_file_blueprint
    # app.register_blueprint(upload_file_blueprint)

    return app
