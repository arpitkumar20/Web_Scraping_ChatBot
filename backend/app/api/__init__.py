from .routes_chat import chat_bp
from .routes_admin import admin_bp
from .routes_wati import wati_bp
# from .routes_scraper import scraper_bp
from .routes_connector import connector_bp
from .routes_scraper_v2 import scraper_bp
from .short_routes import shorter_routes
from .extract_document import extract_documents
from .zoho_data import zoho_bp

def init_routes(app):
    app.register_blueprint(chat_bp, url_prefix="/chat")
    app.register_blueprint(admin_bp, url_prefix="/admin")
    app.register_blueprint(wati_bp, url_prefix="/wati")
    app.register_blueprint(scraper_bp, url_prefix="/scrap")
    app.register_blueprint(connector_bp, url_prefix="/connector")
    # app.register_blueprint(scraper_db, url_prefix="/db")
    app.register_blueprint(shorter_routes, url_prefix="/routes")
    app.register_blueprint(extract_documents, url_prefix="/documents")
    app.register_blueprint(zoho_bp, url_prefix="/zoho")