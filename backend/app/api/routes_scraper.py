# import logging
# import uuid
# from pathlib import Path
# from flask import Blueprint, jsonify, request
# from threading import Thread, Lock
# from app.helper.utils import COMMON
# from app.web_scraping.scraper_function import web_scraping
# from app.services.embeddings_store_v2 import store_embeddings_from_folder
# from app.models.postgresql_db import PostgreSQL

# scraper_bp = Blueprint("scrap", __name__)
# lock = Lock()  # For thread-safe list updates

# # In-memory job tracker {job_id: {"status": "...", "results": [...]}}
# job_status = {}

# def async_store_embeddings(result_path, web_info, job_id):
#     """
#     Stores embeddings and updates job status.
#     """
#     try:
#         # Store embeddings
#         store_data = store_embeddings_from_folder(str(result_path.parent))

#         # Normalize namespace
#         namespace_value = store_data.get("namespace") if isinstance(store_data, dict) else str(store_data)

#         # Add namespace to web_info
#         web_info_with_ns = dict(web_info)
#         web_info_with_ns["namespace"] = namespace_value

#         # Save JSON via COMMON helper
#         COMMON.save_json_data(web_info_with_ns)
#         logging.info("Website record saved successfully")

#         with lock:
#             job_status[job_id]["results"].append({
#                 "message": "Scraping and embedding storing completed.",
#                 "url": web_info.get("url", "unknown"),
#                 "namespace": namespace_value
#             })
#             job_status[job_id]["status"] = "completed"

#     except Exception as e:
#         logging.exception("Error storing embeddings for %s", web_info.get("url", "unknown"))
#         with lock:
#             job_status[job_id]["results"].append({
#                 "message": "Failed to store embeddings",
#                 "url": web_info.get("url", "unknown"),
#                 "error": str(e)
#             })
#             job_status[job_id]["status"] = "failed"


# def background_scraping(site_urls, job_id):
#     """
#     Runs scraping and embedding storage in background threads.
#     """
#     threads = []
#     for url in site_urls:
#         try:
#             web_info, result_path = web_scraping(url)
#             if result_path is None:
#                 raise Exception("Failed to generate embedding data.")

#             thread = Thread(
#                 target=async_store_embeddings,
#                 args=(result_path, web_info, job_id)
#             )
#             thread.start()
#             threads.append(thread)

#         except Exception as e:
#             logging.exception("Error processing URL %s", url)
#             with lock:
#                 job_status[job_id]["results"].append({
#                     "message": "Failed to scrape data",
#                     "url": url,
#                     "error": str(e)
#                 })
#                 job_status[job_id]["status"] = "failed"

#     logging.info("Background scraping started for all URLs.")


# @scraper_bp.route("/web-scraper", methods=["POST"])
# def scraper():
#     data = request.json or {}
#     site_urls = data.get("urls")

#     if not site_urls or not isinstance(site_urls, list):
#         return jsonify({"error": "'urls' field missing or invalid"}), 400

#     # Create a job ID to track progress
#     job_id = str(uuid.uuid4())
#     with lock:
#         job_status[job_id] = {"status": "in-progress", "results": []}

#     # Start background thread for scraping
#     Thread(target=background_scraping, args=(site_urls, job_id), daemon=True).start()

#     logging.info("Inserting result into web scraping table.")
#     PostgreSQL().insert_web_scraping_status({
#         "status": "success",
#         "message": "Scraping started in background.",
#         "job_id": job_id
#     })

#     # Immediately return response with job_id
#     return jsonify({
#         "status": "success",
#         "message": "Scraping started in background.",
#         "job_id": job_id
#     })


# @scraper_bp.route("/web-scraper/status/<job_id>", methods=["GET"])
# def scraper_status(job_id):
#     """
#     Endpoint to check job status and results.
#     Updates the job status in PostgreSQL for the same job_id.
#     """
#     with lock:
#         if job_id not in job_status:
#             return jsonify({"error": "Invalid job_id"}), 404

#         job_data = {
#             "job_id": job_id,
#             "status": job_status[job_id]["status"],
#             "message": job_status[job_id].get("results", [{}])[0].get("message") if job_status[job_id].get("results") else None,
#             "namespace": job_status[job_id].get("results", [{}])[0].get("namespace") if job_status[job_id].get("results") else None,
#             "url": job_status[job_id].get("results", [{}])[0].get("url") if job_status[job_id].get("results") else None,
#         }

#         PostgreSQL().update_web_scraping_status(job_data)

#         return jsonify({
#             "job_id": job_id,
#             "status": job_status[job_id]["status"],
#             "results": job_status[job_id]["results"]
#         })


# ====================================================================================================================
# BASED ON ONE URL :
# ====================================================================================================================

# import logging
# import uuid
# from pathlib import Path
# from flask import Blueprint, jsonify, request
# from threading import Thread, Lock
# from app.helper.utils import COMMON
# from app.web_scraping.scraper_function import web_scraping
# from app.services.embeddings_store_v2 import store_embeddings_from_folder
# from app.models.postgresql_db import PostgreSQL

# scraper_bp = Blueprint("scrap", __name__)
# lock = Lock()

# # Job tracker {job_id: {"scraping_status": "...", "embedding_status": "...", "results": [...], "step": "scraping/embedding/done"}}
# job_status = {}

# # ------------------ BACKGROUND TASKS ------------------

# def background_scraping(site_urls, job_id):
#     try:
#         with lock:
#             job_status[job_id]["scraping_status"] = "in-progress"
#             job_status[job_id]["step"] = "scraping"

#         results = []
#         for url in site_urls:
#             try:
#                 web_info, result_path = web_scraping(url)
#                 if result_path is None:
#                     raise Exception("Failed to scrape content")

#                 results.append({
#                     "url": url,
#                     "result_path": str(result_path),
#                     "web_info": web_info
#                 })

#             except Exception as e:
#                 logging.exception("Error scraping %s", url)
#                 results.append({"url": url, "error": str(e)})

#         with lock:
#             job_status[job_id]["scraping_status"] = "completed"
#             job_status[job_id]["results"] = results
#             job_status[job_id]["step"] = "waiting-for-embedding"

#         PostgreSQL().update_web_scraping_status({
#             "job_id": job_id,
#             "status": "scraping_completed"
#         })

#     except Exception as e:
#         logging.exception("Scraping failed for job %s", job_id)
#         with lock:
#             job_status[job_id]["scraping_status"] = "failed"
#             job_status[job_id]["step"] = "failed"
#         PostgreSQL().update_web_scraping_status({
#             "job_id": job_id,
#             "status": "scraping_failed",
#             "error": str(e)
#         })


# def background_embedding(job_id):
#     try:
#         with lock:
#             if "results" not in job_status[job_id] or not job_status[job_id]["results"]:
#                 raise Exception("No scraping results found. Run scraping first.")
#             job_status[job_id]["embedding_status"] = "in-progress"
#             job_status[job_id]["step"] = "embedding"

#         updated_results = []
#         for res in job_status[job_id]["results"]:
#             if "error" in res:
#                 updated_results.append(res)
#                 continue

#             try:
#                 result_path = Path(res["result_path"])
#                 web_info = res["web_info"]

#                 store_data = store_embeddings_from_folder(str(result_path.parent))
#                 namespace_value = store_data.get("namespace") if isinstance(store_data, dict) else str(store_data)

#                 web_info_with_ns = dict(web_info)
#                 web_info_with_ns["namespace"] = namespace_value
#                 COMMON.save_json_data(web_info_with_ns)

#                 updated_results.append({
#                     "url": res["url"],
#                     "namespace": namespace_value,
#                     "message": "Embedding stored successfully"
#                 })
#             except Exception as e:
#                 logging.exception("Error embedding %s", res.get("url"))
#                 updated_results.append({
#                     "url": res.get("url"),
#                     "error": str(e)
#                 })

#         with lock:
#             job_status[job_id]["embedding_status"] = "completed"
#             job_status[job_id]["step"] = "done"
#             job_status[job_id]["results"] = updated_results

#         PostgreSQL().update_web_scraping_status({
#             "job_id": job_id,
#             "status": "embedding_completed"
#         })

#     except Exception as e:
#         logging.exception("Embedding failed for job %s", job_id)
#         with lock:
#             job_status[job_id]["embedding_status"] = "failed"
#             job_status[job_id]["step"] = "failed"
#         PostgreSQL().update_web_scraping_status({
#             "job_id": job_id,
#             "status": "embedding_failed",
#             "error": str(e)
#         })


# # ------------------ ROUTES ------------------

# @scraper_bp.route("/web-scraper", methods=["POST"])
# def start_scraper():
#     data = request.json or {}
#     site_urls = data.get("urls")

#     if not site_urls or not isinstance(site_urls, list):
#         return jsonify({"error": "'urls' field missing or invalid"}), 400

#     job_id = str(uuid.uuid4())
#     with lock:
#         job_status[job_id] = {
#             "job_id": job_id,
#             "scraping_status": "queued",
#             "embedding_status": "pending",
#             "step": "queued",
#             "results": []
#         }

#     Thread(target=background_scraping, args=(site_urls, job_id), daemon=True).start()

#     PostgreSQL().insert_web_scraping_status({
#         "job_id": job_id,
#         "status": "scraping_started",
#         "message": "Scraping started in background."
#     })

#     return jsonify({
#         "status": "success",
#         "message": "Scraping started in background.",
#         "job_id": job_id
#     })


# @scraper_bp.route("/embedding/<job_id>", methods=["POST"])
# def start_embedding(job_id):
#     with lock:
#         if job_id not in job_status:
#             return jsonify({"error": "Invalid job_id"}), 404
#         if job_status[job_id]["scraping_status"] != "completed":
#             return jsonify({"error": "Scraping not completed yet."}), 400

#         job_status[job_id]["embedding_status"] = "queued"
#         job_status[job_id]["step"] = "embedding_queued"

#     Thread(target=background_embedding, args=(job_id,), daemon=True).start()

#     PostgreSQL().insert_web_scraping_status({
#         "job_id": job_id,
#         "status": "embedding_started",
#         "message": "Embedding started in background."
#     })

#     return jsonify({
#         "status": "success",
#         "message": "Embedding started in background.",
#         "job_id": job_id
#     })


# @scraper_bp.route("/job-status/<job_id>", methods=["GET"])
# def job_status_api(job_id):
#     with lock:
#         if job_id not in job_status:
#             return jsonify({"error": "Invalid job_id"}), 404

#         return jsonify(job_status[job_id])





# ====================================================================================================================
# BASED ON MULTIPLE URL :
# ====================================================================================================================



import uuid
from pathlib import Path
from app.core.logging import get_logger
from flask import Blueprint, jsonify, request
from threading import Thread, Lock
from app.helper.utils import COMMON
from app.web_scraping.scraper_function import web_scraping
from app.services.embeddings_store_v2 import store_embeddings_from_folder
from app.models.postgresql_db import PostgreSQL

scraper_bp = Blueprint("scrap", __name__)
logger = get_logger(__name__)
lock = Lock()

# Job tracker {job_id: {...}}
job_status = {}

# ------------------ BACKGROUND TASKS ------------------

def background_scraping(url, job_id):
    try:
        with lock:
            job_status[job_id]["scraping_status"] = "in-progress"
            job_status[job_id]["step"] = "scraping"

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

        # Modified PostgreSQL insert/update
        job_data = {
            "job_id": job_id,
            "status": job_status[job_id]["scraping_status"],
            "message": job_status[job_id].get("results", [{}])[0].get("message") 
                       if job_status[job_id].get("results") else None,
            "namespace": job_status[job_id].get("results", [{}])[0].get("namespace") 
                         if job_status[job_id].get("results") else None,
            "url": job_status[job_id].get("results", [{}])[0].get("url") 
                   if job_status[job_id].get("results") else None,
        }

        PostgreSQL().update_web_scraping_status(job_data)

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

        # Modified PostgreSQL update
        job_data = {
            "job_id": job_id,
            "status": job_status[job_id]["embedding_status"],
            "message": job_status[job_id].get("results", [{}])[0].get("message") 
                       if job_status[job_id].get("results") else None,
            "namespace": job_status[job_id].get("results", [{}])[0].get("namespace") 
                         if job_status[job_id].get("results") else None,
            "url": job_status[job_id].get("results", [{}])[0].get("url") 
                   if job_status[job_id].get("results") else None,
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
            "url": job_status[job_id].get("results", [{}])[0].get("url") 
                   if job_status[job_id].get("results") else None,
        }
        PostgreSQL().update_web_scraping_status(job_data)


# ------------------ ROUTES ------------------

@scraper_bp.route("/web-scraper", methods=["POST"])
def start_scraper():
    data = request.json or {}
    site_urls = data.get("urls")

    if not site_urls or not isinstance(site_urls, list):
        return jsonify({"error": "'urls' field missing or invalid"}), 400

    response_jobs = []

    for url in site_urls:
        job_id = str(uuid.uuid4())
        with lock:
            job_status[job_id] = {
                "job_id": job_id,
                "scraping_status": "queued",
                "embedding_status": "pending",
                "step": "queued",
                "results": []
            }

        Thread(target=background_scraping, args=(url, job_id), daemon=True).start()

        # Modified PostgreSQL insert for scraping start
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


@scraper_bp.route("/embedding/<job_id>", methods=["POST"])
def start_embedding(job_id):
    with lock:
        if job_id not in job_status:
            return jsonify({"error": "Invalid job_id"}), 404
        if job_status[job_id]["scraping_status"] != "completed":
            return jsonify({"error": "Scraping not completed yet."}), 400

        job_status[job_id]["embedding_status"] = "queued"
        job_status[job_id]["step"] = "embedding_queued"

    Thread(target=background_embedding, args=(job_id,), daemon=True).start()

    PostgreSQL().insert_web_scraping_status({
        "status": "success",
        "message": "Embedding started in background.",
        "job_id": job_id
    })

    return jsonify({
        "status": "success",
        "message": "Embedding started in background.",
        "job_id": job_id
    })


@scraper_bp.route("/job-status/<job_id>", methods=["GET"])
def job_status_api(job_id):
    with lock:
        if job_id not in job_status:
            return jsonify({"error": "Invalid job_id"}), 404
        return jsonify(job_status[job_id])


@scraper_bp.route("/embedding_database", methods=["POST"])
def start_database_embedding():
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

            Thread(target=background_scraping, args=(url.get('url'), job_id), daemon=True).start()

            # Modified PostgreSQL insert for scraping start
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

