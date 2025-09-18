from .routes_chat import chat_bp
from .routes_admin import admin_bp
from .routes_wati import wati_bp
from .routes_scraper import scraper_bp
from .routes_database import database_bp


def init_routes(app):
    app.register_blueprint(chat_bp, url_prefix="/chat")
    app.register_blueprint(admin_bp, url_prefix="/admin")
    app.register_blueprint(wati_bp, url_prefix="/wati")
    app.register_blueprint(scraper_bp, url_prefix="/scrap")
    app.register_blueprint(database_bp, url_prefix="/database")