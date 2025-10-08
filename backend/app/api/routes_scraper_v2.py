
# import uuid
# from pathlib import Path
# from threading import Thread, Lock
# from flask import Blueprint, jsonify, request
# from app.core.logging import get_logger

# from app.helper.utils import COMMON
# from app.web_scraping.scraper_function import web_scraping
# from app.services.embeddings_store_v2 import store_embeddings_from_folder
# from app.models.postgresql_db import PostgreSQL

# scraper_bp = Blueprint("scrap", __name__)
# lock = Lock()
# job_status = {}

# logger = get_logger(__name__)

# @scraper_bp.route("/job-status/<job_id>", methods=["GET"])
# def job_status_api(job_id):
#     """Fetch the current status of a scraping/embedding job"""
#     with lock:
#         if job_id not in job_status:
#             return jsonify({"error": "Invalid job_id"}), 404
#         return jsonify(job_status[job_id])

# @scraper_bp.route("/web-scraper", methods=["POST"])
# def start_scraper():
#     """
#     Start scraping and embedding workflow for URLs provided in the request body.
#     """
#     try:
#         data = request.json or {}
#         site_urls = data.get("urls")
#         company_name = data.get("company_name")

#         if not site_urls or not isinstance(site_urls, list):
#             return jsonify({"error": "'urls' field missing or invalid"}), 400

#         response_jobs = []

#         for url in site_urls:
#             job_id = str(uuid.uuid4())
#             with lock:
#                 job_status[job_id] = {
#                     "job_id": job_id,
#                     "scraping_status": "queued",
#                     "embedding_status": "pending",
#                     "step": "queued",
#                     "results": []
#                 }

#             # Start scraping in a background thread
#             Thread(target=background_scraping, args=(company_name, url, job_id), daemon=True).start()

#             # Insert initial scraping status in PostgreSQL
#             PostgreSQL().insert_web_scraping_status({
#                 "status": "success",
#                 "message": "Scraping started in background.",
#                 "job_id": job_id
#             })

#             response_jobs.append({"job_id": job_id, "url": url})

#         return jsonify({
#             "status": "success",
#             "message": "Scraping started for multiple URLs.",
#             "jobs": response_jobs
#         })

#     except Exception as e:
#         logger.error(f"Error in embedding database API: {e}")
#         return jsonify({"error": str(e)}), 500


# def background_scraping(company_name, url, job_id):
#     """
#     Scrape a URL and automatically trigger embedding after completion.
#     """
#     try:
#         with lock:
#             job_status[job_id]["scraping_status"] = "in-progress"
#             job_status[job_id]["step"] = "scraping"

#         # Perform web scraping
#         web_info, result_path = web_scraping(url)
#         if result_path is None:
#             raise Exception("Failed to scrape content")

#         with lock:
#             job_status[job_id]["scraping_status"] = "completed"
#             job_status[job_id]["results"] = [{
#                 "url": url,
#                 "result_path": str(result_path),
#                 "web_info": web_info
#             }]
#             job_status[job_id]["step"] = "waiting-for-embedding"

#         # Update PostgreSQL status
#         job_data = {
#             "job_id": job_id,
#             "status": job_status[job_id]["scraping_status"],
#             "message": job_status[job_id].get("results", [{}])[0].get("message"),
#             "namespace": job_status[job_id].get("results", [{}])[0].get("namespace"),
#             "url": job_status[job_id].get("results", [{}])[0].get("url"),
#         }
#         PostgreSQL().update_web_scraping_status(job_data)

#         # Automatically start embedding
#         Thread(target=background_embedding, args=(company_name, job_id,), daemon=True).start()

#     except Exception as e:
#         logger.exception("Scraping failed for job %s", job_id)
#         with lock:
#             job_status[job_id]["scraping_status"] = "failed"
#             job_status[job_id]["step"] = "failed"
#             job_status[job_id]["results"] = [{"url": url, "error": str(e)}]

#         job_data = {
#             "job_id": job_id,
#             "status": "failed",
#             "message": str(e),
#             "namespace": None,
#             "url": url,
#         }
#         PostgreSQL().update_web_scraping_status(job_data)


# def background_embedding(company_name, job_id):
#     """
#     Generate embeddings for scraped content and update PostgreSQL.
#     """
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

#                 store_data = store_embeddings_from_folder(company_name, str(result_path.parent))
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
#                 logger.exception("Error embedding %s", res.get("url"))
#                 updated_results.append({
#                     "url": res.get("url"),
#                     "error": str(e)
#                 })

#         with lock:
#             job_status[job_id]["embedding_status"] = "completed"
#             job_status[job_id]["step"] = "done"
#             job_status[job_id]["results"] = updated_results

#         # Update PostgreSQL
#         job_data = {
#             "job_id": job_id,
#             "status": job_status[job_id]["embedding_status"],
#             "message": job_status[job_id].get("results", [{}])[0].get("message"),
#             "namespace": job_status[job_id].get("results", [{}])[0].get("namespace"),
#             "url": job_status[job_id].get("results", [{}])[0].get("url"),
#         }
#         PostgreSQL().update_web_scraping_status(job_data)

#     except Exception as e:
#         logger.exception("Embedding failed for job %s", job_id)
#         with lock:
#             job_status[job_id]["embedding_status"] = "failed"
#             job_status[job_id]["step"] = "failed"

#         job_data = {
#             "job_id": job_id,
#             "status": "failed",
#             "message": str(e),
#             "namespace": None,
#             "url": job_status[job_id].get("results", [{}])[0].get("url"),
#         }
#         PostgreSQL().update_web_scraping_status(job_data)




import uuid
from pathlib import Path
from threading import Thread, Lock
from flask import Blueprint, jsonify, request
from app.core.logging import get_logger

from app.helper.utils import COMMON
from app.web_scraping.scraper_function import web_scraping
from app.services.embeddings_store_v2 import store_embeddings_from_folder
from app.models.postgresql_db import PostgreSQL

scraper_bp = Blueprint("scrap", __name__)
lock = Lock()
job_status = {}

logger = get_logger(__name__)

@scraper_bp.route("/job-status/<job_id>", methods=["GET"])
def job_status_api(job_id):
    """Fetch the current status of a scraping/embedding job"""
    with lock:
        if job_id not in job_status:
            return jsonify({"error": "Invalid job_id"}), 404
        return jsonify(job_status[job_id])

@scraper_bp.route("/web-scraper", methods=["POST"])
def start_scraper():
    """
    Start scraping and embedding workflow for URLs provided in the request body.
    """
    try:
        data = request.json or {}
        site_urls = data.get("urls")
        company_name = data.get("company_name")

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

            # Start scraping in a background thread
            Thread(target=background_scraping, args=(company_name, url, job_id), daemon=True).start()

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


# def background_scraping(company_name, url, job_id):
#     """
#     Scrape a URL and automatically trigger embedding after completion.
#     """
#     try:
#         with lock:
#             job_status[job_id]["scraping_status"] = "in-progress"
#             job_status[job_id]["step"] = "scraping"

#         # Perform web scraping
#         # NOTE: ignore any 'web_info' returned by web_scraping and only use result_path
#         _, result_path = web_scraping(url)
#         if result_path is None:
#             raise Exception("Failed to scrape content")

#         with lock:
#             job_status[job_id]["scraping_status"] = "completed"
#             # removed web_info storage as requested
#             job_status[job_id]["results"] = [{
#                 "url": url,
#                 "result_path": str(result_path)
#             }]
#             job_status[job_id]["step"] = "waiting-for-embedding"

#         # Update PostgreSQL status
#         job_data = {
#             "job_id": job_id,
#             "status": job_status[job_id]["scraping_status"],
#             "message": job_status[job_id].get("results", [{}])[0].get("message"),
#             "namespace": job_status[job_id].get("results", [{}])[0].get("namespace"),
#             "url": job_status[job_id].get("results", [{}])[0].get("url"),
#         }
#         PostgreSQL().update_web_scraping_status(job_data)

#         # Automatically start embedding
#         Thread(target=background_embedding, args=(company_name, job_id,), daemon=True).start()

#     except Exception as e:
#         logger.exception("Scraping failed for job %s", job_id)
#         with lock:
#             job_status[job_id]["scraping_status"] = "failed"
#             job_status[job_id]["step"] = "failed"
#             job_status[job_id]["results"] = [{"url": url, "error": str(e)}]

#         job_data = {
#             "job_id": job_id,
#             "status": "failed",
#             "message": str(e),
#             "namespace": None,
#             "url": url,
#         }
#         PostgreSQL().update_web_scraping_status(job_data)



def background_scraping(company_name, url, job_id):
    """
    Scrape a URL and automatically trigger embedding after completion.
    """
    try:
        with lock:
            job_status[job_id]["scraping_status"] = "in-progress"
            job_status[job_id]["step"] = "scraping"

        # Perform web scraping (now returns only a single result_path)
        result_path = web_scraping(url)
        if result_path is None:
            raise Exception("Failed to scrape content")

        with lock:
            job_status[job_id]["scraping_status"] = "completed"
            job_status[job_id]["results"] = [{
                "url": url,
                "result_path": str(result_path)
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
        Thread(target=background_embedding, args=(company_name, job_id,), daemon=True).start()

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


def background_embedding(company_name, job_id):
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

                # generate embeddings from scraped folder
                store_data = store_embeddings_from_folder(company_name, str(result_path.parent))
                namespace_value = store_data.get("namespace") if isinstance(store_data, dict) else str(store_data)

                # Instead of using web_info and copying it, save a compact JSON with url/result_path/namespace.
                saved_json = {
                    "url": res.get("url"),
                    "result_path": str(result_path),
                    "namespace": namespace_value
                }
                COMMON.save_json_data(saved_json)

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
