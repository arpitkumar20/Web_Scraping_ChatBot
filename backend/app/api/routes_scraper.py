import logging
import uuid
from pathlib import Path
from flask import Blueprint, jsonify, request
from threading import Thread, Lock
from app.helper.utils import COMMON
from app.web_scraping.scraper_function import web_scraping
from app.services.embeddings_store_v2 import store_embeddings_from_folder
from app.models.postgresql_db import PostgreSQL

scraper_bp = Blueprint("scrap", __name__)
lock = Lock()  # For thread-safe list updates

# In-memory job tracker {job_id: {"status": "...", "results": [...]}}
job_status = {}

def async_store_embeddings(result_path, web_info, job_id):
    """
    Stores embeddings and updates job status.
    """
    try:
        # Store embeddings
        store_data = store_embeddings_from_folder(str(result_path.parent))

        # Normalize namespace
        namespace_value = store_data.get("namespace") if isinstance(store_data, dict) else str(store_data)

        # Add namespace to web_info
        web_info_with_ns = dict(web_info)
        web_info_with_ns["namespace"] = namespace_value

        # Save JSON via COMMON helper
        COMMON.save_json_data(web_info_with_ns)
        logging.info("Website record saved successfully")

        with lock:
            job_status[job_id]["results"].append({
                "message": "Scraping and embedding storing completed.",
                "url": web_info.get("url", "unknown"),
                "namespace": namespace_value
            })
            job_status[job_id]["status"] = "completed"

    except Exception as e:
        logging.exception("Error storing embeddings for %s", web_info.get("url", "unknown"))
        with lock:
            job_status[job_id]["results"].append({
                "message": "Failed to store embeddings",
                "url": web_info.get("url", "unknown"),
                "error": str(e)
            })
            job_status[job_id]["status"] = "failed"


def background_scraping(site_urls, job_id):
    """
    Runs scraping and embedding storage in background threads.
    """
    threads = []
    for url in site_urls:
        try:
            web_info, result_path = web_scraping(url)
            if result_path is None:
                raise Exception("Failed to generate embedding data.")

            thread = Thread(
                target=async_store_embeddings,
                args=(result_path, web_info, job_id)
            )
            thread.start()
            threads.append(thread)

        except Exception as e:
            logging.exception("Error processing URL %s", url)
            with lock:
                job_status[job_id]["results"].append({
                    "message": "Failed to scrape data",
                    "url": url,
                    "error": str(e)
                })
                job_status[job_id]["status"] = "failed"

    logging.info("Background scraping started for all URLs.")


@scraper_bp.route("/web-scraper", methods=["POST"])
def scraper():
    data = request.json or {}
    site_urls = data.get("urls")

    if not site_urls or not isinstance(site_urls, list):
        return jsonify({"error": "'urls' field missing or invalid"}), 400

    # Create a job ID to track progress
    job_id = str(uuid.uuid4())
    with lock:
        job_status[job_id] = {"status": "in-progress", "results": []}

    # Start background thread for scraping
    Thread(target=background_scraping, args=(site_urls, job_id), daemon=True).start()

    logging.info("Inserting result into web scraping table.")
    PostgreSQL().insert_web_scraping_status({
        "status": "success",
        "message": "Scraping started in background.",
        "job_id": job_id
    })

    # Immediately return response with job_id
    return jsonify({
        "status": "success",
        "message": "Scraping started in background.",
        "job_id": job_id
    })


@scraper_bp.route("/web-scraper/status/<job_id>", methods=["GET"])
def scraper_status(job_id):
    """
    Endpoint to check job status and results.
    Updates the job status in PostgreSQL for the same job_id.
    """
    with lock:
        if job_id not in job_status:
            return jsonify({"error": "Invalid job_id"}), 404

        job_data = {
            "job_id": job_id,
            "status": job_status[job_id]["status"],
            "message": job_status[job_id].get("results", [{}])[0].get("message") if job_status[job_id].get("results") else None,
            "namespace": job_status[job_id].get("results", [{}])[0].get("namespace") if job_status[job_id].get("results") else None,
            "url": job_status[job_id].get("results", [{}])[0].get("url") if job_status[job_id].get("results") else None,
        }

        PostgreSQL().update_web_scraping_status(job_data)

        return jsonify({
            "job_id": job_id,
            "status": job_status[job_id]["status"],
            "results": job_status[job_id]["results"]
        })
