
import uuid
from pathlib import Path
from threading import Thread, Lock
from flask import Blueprint, jsonify, request
from app.core.logging import get_logger

from app.helper.utils import COMMON
from app.web_scraping.scraper_function import web_scraping
from app.services.embeddings_store_v2 import store_embeddings_from_folder
from app.models.postgresql_db import PostgreSQL

scraper_db = Blueprint("db", __name__)
lock = Lock()
job_status = {}

logger = get_logger(__name__)

@scraper_db.route("/job-status/<job_id>", methods=["GET"])
def job_status_api(job_id):
    """Fetch the current status of a scraping/embedding job"""
    with lock:
        if job_id not in job_status:
            return jsonify({"error": "Invalid job_id"}), 404
        return jsonify(job_status[job_id])


@scraper_db.route("/embedding_database", methods=["POST"])
def start_database_embedding():
    """
    Start scraping and embedding workflow for URLs fetched from the database.
    """
    try:
        data = request.json
        connector_type = data.get('connector_type')  # Required: 'mysql' or 'postgresql'
        schema_name = data.get('schema_name')
        table_and_columns = data.get('table_and_columns')  # [{'table_name': 'users', 'columns': ['id', 'name']}]
        query = data.get('query')
        primary_column = data.get('primary_column')
        offset = data.get('offset', 0)
        limit = data.get('limit', 100)

        if not connector_type:
            return jsonify({"error": "connector_type is required"}), 400

        if not (schema_name and table_and_columns) and not query:
            return jsonify({"error": "Either (schema_name and table_and_columns) or query is required"}), 400

        # Initialize DB instance dynamically
        if connector_type == 'mysql':
            # Placeholder if MySQL support is added
            pass
        elif connector_type == 'postgresql':
            db_instance = PostgreSQL()
        else:
            return jsonify({"error": f"Unsupported connector_type '{connector_type}'"}), 400

        # Fetch rows from DB
        fetch_rows = db_instance.scan_row(
            schema_name=schema_name,
            table_and_columns=table_and_columns,
            query=query,
            primary_column=primary_column,
            offset=offset,
            limit=limit
        )
        logger.info(f"Fetched {len(fetch_rows)} rows from database.")

        response_jobs = []
        for url in fetch_rows:
            job_id = str(uuid.uuid4())
            with lock:
                job_status[job_id] = {
                    "job_id": job_id,
                    "scraping_status": "queued",
                    "embedding_status": "pending",
                    "step": "queued",
                    "results": []
                }

            # Start scraping in a background thread
            Thread(target=background_scraping, args=(url.get('url'), job_id), daemon=True).start()

            # Insert initial scraping status in PostgreSQL
            PostgreSQL().insert_web_scraping_status({
                "status": "success",
                "message": "Scraping started in background.",
                "job_id": job_id
            })

            response_jobs.append({"job_id": job_id, "url": url})

        return jsonify({
            "status": "success",
            "message": "Scraping started for multiple URLs.",
            "jobs": response_jobs
        })

    except Exception as e:
        logger.error(f"Error in embedding database API: {e}")
        return jsonify({"error": str(e)}), 500


def background_scraping(url, job_id):
    """
    Scrape a URL and automatically trigger embedding after completion.
    """
    try:
        with lock:
            job_status[job_id]["scraping_status"] = "in-progress"
            job_status[job_id]["step"] = "scraping"

        # Perform web scraping
        web_info, result_path = web_scraping(url)
        if result_path is None:
            raise Exception("Failed to scrape content")

        with lock:
            job_status[job_id]["scraping_status"] = "completed"
            job_status[job_id]["results"] = [{
                "url": url,
                "result_path": str(result_path),
                "web_info": web_info
            }]
            job_status[job_id]["step"] = "waiting-for-embedding"

        # Update PostgreSQL status
        job_data = {
            "job_id": job_id,
            "status": job_status[job_id]["scraping_status"],
            "message": job_status[job_id].get("results", [{}])[0].get("message"),
            "namespace": job_status[job_id].get("results", [{}])[0].get("namespace"),
            "url": job_status[job_id].get("results", [{}])[0].get("url"),
        }
        PostgreSQL().update_web_scraping_status(job_data)

        # Automatically start embedding
        Thread(target=background_embedding, args=(job_id,), daemon=True).start()

    except Exception as e:
        logger.exception("Scraping failed for job %s", job_id)
        with lock:
            job_status[job_id]["scraping_status"] = "failed"
            job_status[job_id]["step"] = "failed"
            job_status[job_id]["results"] = [{"url": url, "error": str(e)}]

        job_data = {
            "job_id": job_id,
            "status": "failed",
            "message": str(e),
            "namespace": None,
            "url": url,
        }
        PostgreSQL().update_web_scraping_status(job_data)


def background_embedding(job_id):
    """
    Generate embeddings for scraped content and update PostgreSQL.
    """
    try:
        with lock:
            if "results" not in job_status[job_id] or not job_status[job_id]["results"]:
                raise Exception("No scraping results found. Run scraping first.")
            job_status[job_id]["embedding_status"] = "in-progress"
            job_status[job_id]["step"] = "embedding"

        updated_results = []
        for res in job_status[job_id]["results"]:
            if "error" in res:
                updated_results.append(res)
                continue

            try:
                result_path = Path(res["result_path"])
                web_info = res["web_info"]

                store_data = store_embeddings_from_folder(str(result_path.parent))
                namespace_value = store_data.get("namespace") if isinstance(store_data, dict) else str(store_data)

                web_info_with_ns = dict(web_info)
                web_info_with_ns["namespace"] = namespace_value
                COMMON.save_json_data(web_info_with_ns)

                updated_results.append({
                    "url": res["url"],
                    "namespace": namespace_value,
                    "message": "Embedding stored successfully"
                })
            except Exception as e:
                logger.exception("Error embedding %s", res.get("url"))
                updated_results.append({
                    "url": res.get("url"),
                    "error": str(e)
                })

        with lock:
            job_status[job_id]["embedding_status"] = "completed"
            job_status[job_id]["step"] = "done"
            job_status[job_id]["results"] = updated_results

        # Update PostgreSQL
        job_data = {
            "job_id": job_id,
            "status": job_status[job_id]["embedding_status"],
            "message": job_status[job_id].get("results", [{}])[0].get("message"),
            "namespace": job_status[job_id].get("results", [{}])[0].get("namespace"),
            "url": job_status[job_id].get("results", [{}])[0].get("url"),
        }
        PostgreSQL().update_web_scraping_status(job_data)

    except Exception as e:
        logger.exception("Embedding failed for job %s", job_id)
        with lock:
            job_status[job_id]["embedding_status"] = "failed"
            job_status[job_id]["step"] = "failed"

        job_data = {
            "job_id": job_id,
            "status": "failed",
            "message": str(e),
            "namespace": None,
            "url": job_status[job_id].get("results", [{}])[0].get("url"),
        }
        PostgreSQL().update_web_scraping_status(job_data)
