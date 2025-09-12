from flask import Blueprint, jsonify, request
from app.web_scraping.scraper_function import web_scraping
 
 
scraper_bp = Blueprint("scrap", __name__)
 
@scraper_bp.route("/web-scraper", methods=["POST"])
def scraper():
    data = request.json or {}
    
    # Expecting a list of URLs in "urls" field
    site_urls = data.get("urls")
    
    if not site_urls or not isinstance(site_urls, list):
        return jsonify({"error": "Missing required fields or 'urls' is not a list"}), 400
    
    results = []
    for url in site_urls:
        try:
            result = web_scraping(url)  # Call your scraping function for each URL
            results.append({"url": url, "data": result})
        except Exception as e:
            # Handle errors for individual URLs
            results.append({"url": url, "error": str(e)})
    
    return jsonify(results)