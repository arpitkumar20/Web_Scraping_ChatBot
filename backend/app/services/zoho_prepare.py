# import json
# import re
# import hashlib
# from typing import Any, Dict, List, Iterable, Tuple, Callable, Optional

# def prepare_and_embed_zoho_rows(
#     zoho_data: Any,
#     embed_fn: Callable[[List[str], List[Dict[str, Any]]], None],
#     *,
#     namespace: str = "zoho-knowledge",
#     id_prefix: str = "zoho",
#     chunk_size: int = 1000,
#     chunk_overlap: int = 120,
#     embed_batch: int = 100,
# ) -> Dict[str, int]:
#     """
#     Normalize Zoho report data from multiple formats, convert each row to text,
#     chunk row-wise, and hand off to `embed_fn(texts, metadatas)` for embedding/upsert.

#     Supported input shapes:
#       A) Applications wrapper:
#          {
#            "applications": [
#              {
#                "owner_name": "...",
#                "app_link_name": "...",
#                "reports": [
#                  {
#                    "report_status": 200,
#                    "report_metadata": {"report_link_name": "Leads", "report_display_name": "Leads"},
#                    "report_data": { "records": { "data": [ {...}, {...} ] } }
#                  }
#                ]
#              }
#            ]
#          }

#       B) Single report payload:
#          {"operation_type":"report","report_link_name":"Leads","records":{"data":[...] } }

#       C) Dict of {report_name: [rows]}  -> {"Leads":[ {...}, {...} ], "Deals":[ {...} ]}
#       D) Plain list of rows:            -> [ {...}, {...} ]
#       E) List of mixed items:           -> [ <A|B|C|D>, ... ]
#     """

#     # -------------------- utilities --------------------
#     def slug(s: str) -> str:
#         s = (s or "").strip()
#         s = re.sub(r"\s+", "-", s)
#         s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
#         return s.lower()

#     def clean_val(v: Any) -> Any:
#         if isinstance(v, str):
#             return " ".join(v.split())
#         return v

#     def to_str(v: Any) -> str:
#         v = clean_val(v)
#         if v is None:
#             return ""
#         if isinstance(v, (list, dict)):
#             return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
#         return str(v)

#     def prefer_display_value(val: Any) -> Any:
#         if isinstance(val, dict) and "display_value" in val:
#             return val.get("display_value")
#         return val

#     def stable_row_id(row: Dict[str, Any]) -> str:
#         for k in ("ID", "id", "row_id", "uuid", "pk", "Zoho_ID", "zoho_id"):
#             if k in row and row[k]:
#                 return str(row[k])
#         payload = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
#         return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

#     def content_hash(row: Dict[str, Any], include: Optional[List[str]] = None, exclude: Optional[List[str]] = None) -> str:
#         if include is not None:
#             keys = [k for k in include if k in row]
#         else:
#             keys = list(row.keys())
#         if exclude:
#             keys = [k for k in keys if k not in exclude]
#         keys = sorted(keys)
#         norm = {k: clean_val(prefer_display_value(row.get(k))) for k in keys}
#         payload = json.dumps(norm, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
#         return hashlib.sha1(payload.encode("utf-8")).hexdigest()

#     def row_to_text(row: Dict[str, Any]) -> str:
#         lines: List[str] = []
#         for key in sorted(row.keys()):
#             val = prefer_display_value(row[key])
#             s = to_str(val)
#             if s != "":
#                 lines.append(f"{key}: {s}")
#         return "\n".join(lines)

#     def split_text(text: str, csize: int, overlap: int) -> List[str]:
#         if len(text) <= csize:
#             return [text]
#         chunks, start = [], 0
#         while start < len(text):
#             end = start + csize
#             chunks.append(text[start:end])
#             if end >= len(text):
#                 break
#             start = max(0, end - overlap)
#         return chunks

#     # -------------------- iterator for many shapes --------------------
#     def iter_reports_and_rows(payload: Any) -> Iterable[Tuple[str, Dict[str, Any], Dict[str, str]]]:
#         # A) applications -> reports -> records.data
#         if isinstance(payload, dict) and isinstance(payload.get("applications"), list):
#             for app in payload["applications"]:
#                 if not isinstance(app, dict):
#                     continue
#                 owner = to_str(app.get("owner_name"))
#                 app_link = to_str(app.get("app_link_name"))
#                 reports = app.get("reports") or []
#                 for rep in reports:
#                     if not isinstance(rep, dict):
#                         continue
#                     if rep.get("report_status") not in (200, "200", None):
#                         continue
#                     md = rep.get("report_metadata") or {}
#                     report_name = to_str(md.get("report_link_name") or md.get("report_display_name") or "")
#                     data = (((rep.get("report_data") or {}).get("records") or {}).get("data"))
#                     if isinstance(data, list):
#                         for r in data:
#                             if isinstance(r, dict):
#                                 yield (report_name, r, {"owner_name": owner, "app_link_name": app_link})
#             return

#         # B) Single report payload
#         if isinstance(payload, dict) and "records" in payload:
#             report_name = to_str(payload.get("report_link_name") or payload.get("report_name") or "")
#             data = (payload.get("records") or {}).get("data")
#             if isinstance(data, list):
#                 for r in data:
#                     if isinstance(r, dict):
#                         yield (report_name, r, {})
#             return

#         # C) Dict of {report_name: [rows]}
#         if isinstance(payload, dict) and "records" not in payload:
#             for k, v in payload.items():
#                 if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], dict)):
#                     for r in v:
#                         if isinstance(r, dict):
#                             yield (to_str(k), r, {})
#             return

#         # D) Plain list of rows
#         if isinstance(payload, list) and (len(payload) == 0 or isinstance(payload[0], dict)):
#             for r in payload:
#                 if isinstance(r, dict):
#                     yield ("", r, {})
#             return

#         # E) List of mixed items
#         if isinstance(payload, list):
#             for item in payload:
#                 yield from iter_reports_and_rows(item)
#             return

#         return

#     # -------------------- main flow --------------------
#     report_count = 0
#     row_count = 0
#     chunk_count = 0

#     buffer_texts: List[str] = []
#     buffer_metas: List[Dict[str, Any]] = []

#     def flush():
#         nonlocal buffer_texts, buffer_metas
#         if buffer_texts:
#             embed_fn(buffer_texts, buffer_metas)
#             buffer_texts, buffer_metas = [], []

#     seen_report_names = set()

#     for report_name, row, ctx in iter_reports_and_rows(zoho_data):
#         if report_name not in seen_report_names:
#             seen_report_names.add(report_name)
#             report_count += 1

#         row_count += 1

#         row_pk = stable_row_id(row)
#         owner = ctx.get("owner_name", "")
#         app = ctx.get("app_link_name", "")

#         owner_s = slug(owner)
#         app_s = slug(app)
#         report_s = slug(report_name) if report_name else "unknown-report"

#         preview = to_str(row.get("Full_Name") or row.get("Name") or row.get("Title") or row_pk)[:120]

#         text = row_to_text(row)
#         chunks = split_text(text, chunk_size, chunk_overlap)
#         chunk_total = len(chunks)
#         chunk_count += chunk_total

#         group_id = f"{id_prefix}:{owner_s}:{app_s}:{report_s}:{row_pk}"
#         row_hash = content_hash(row)

#         for i, ch in enumerate(chunks, 1):
#             vector_id = f"{group_id}:attributes:{i}"
#             meta = {
#                 "source": "zoho",
#                 "namespace": namespace,
#                 "owner_name": owner,
#                 "app_link_name": app,
#                 "report": report_name,
#                 "row_pk": row_pk,
#                 "group_id": group_id,
#                 "field": "attributes",
#                 "chunk_no": i,
#                 "chunk_total": chunk_total,
#                 # 🔽 new line — store full embedded text here
#                 "text_content": ch,
#                 "preview": preview,
#                 "content_hash": row_hash,
#                 "vector_id": vector_id,
#             }
#             buffer_texts.append(ch)
#             buffer_metas.append(meta)

#             if len(buffer_texts) >= embed_batch:
#                 flush()

#     flush()
#     return {"reports": report_count, "rows": row_count, "chunks": chunk_count}





import json
import re
import hashlib
from typing import Any, Dict, List, Iterable, Tuple, Callable, Optional

from app.helper.utils import COMMON

def prepare_and_embed_zoho_rows(
    zoho_data: Any,
    embed_fn: Callable[[List[str], List[Dict[str, Any]]], None],
    *,
    namespace: str = "zoho-knowledge",
    id_prefix: str = "zoho",
    chunk_size: int = 1000,
    chunk_overlap: int = 120,
    embed_batch: int = 100,
    job_id: str = None
) -> Dict[str, int]:
    print("[INFO] Starting prepare_and_embed_zoho_rows...")

    # -------------------- utilities --------------------
    def slug(s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"\s+", "-", s)
        s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
        return s.lower()

    def clean_val(v: Any) -> Any:
        if isinstance(v, str):
            return " ".join(v.split())
        return v

    def to_str(v: Any) -> str:
        v = clean_val(v)
        if v is None:
            return ""
        if isinstance(v, (list, dict)):
            return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return str(v)

    def prefer_display_value(val: Any) -> Any:
        if isinstance(val, dict) and "display_value" in val:
            return val.get("display_value")
        return val

    def stable_row_id(row: Dict[str, Any]) -> str:
        for k in ("ID", "id", "row_id", "uuid", "pk", "Zoho_ID", "zoho_id"):
            if k in row and row[k]:
                return str(row[k])
        payload = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def content_hash(row: Dict[str, Any], include: Optional[List[str]] = None, exclude: Optional[List[str]] = None) -> str:
        if include is not None:
            keys = [k for k in include if k in row]
        else:
            keys = list(row.keys())
        if exclude:
            keys = [k for k in keys if k not in exclude]
        keys = sorted(keys)
        norm = {k: clean_val(prefer_display_value(row.get(k))) for k in keys}
        payload = json.dumps(norm, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def row_to_text(row: Dict[str, Any]) -> str:
        lines: List[str] = []
        for key in sorted(row.keys()):
            val = prefer_display_value(row[key])
            s = to_str(val)
            if s != "":
                lines.append(f"{key}: {s}")
        return "\n".join(lines)

    def split_text(text: str, csize: int, overlap: int) -> List[str]:
        if len(text) <= csize:
            return [text]
        chunks, start = [], 0
        while start < len(text):
            end = start + csize
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = max(0, end - overlap)
        return chunks

    # -------------------- iterator for many shapes --------------------
    def iter_reports_and_rows(payload: Any) -> Iterable[Tuple[str, Dict[str, Any], Dict[str, str]]]:
        if isinstance(payload, dict) and isinstance(payload.get("applications"), list):
            for app in payload["applications"]:
                if not isinstance(app, dict):
                    continue
                owner = to_str(app.get("owner_name"))
                app_link = to_str(app.get("app_link_name"))
                reports = app.get("reports") or []
                for rep in reports:
                    if not isinstance(rep, dict):
                        continue
                    if rep.get("report_status") not in (200, "200", None):
                        continue
                    md = rep.get("report_metadata") or {}
                    report_name = to_str(md.get("report_link_name") or md.get("report_display_name") or "")
                    data = (((rep.get("report_data") or {}).get("records") or {}).get("data"))
                    if isinstance(data, list):
                        for r in data:
                            if isinstance(r, dict):
                                yield (report_name, r, {"owner_name": owner, "app_link_name": app_link})
            return

        if isinstance(payload, dict) and "records" in payload:
            report_name = to_str(payload.get("report_link_name") or payload.get("report_name") or "")
            data = (payload.get("records") or {}).get("data")
            if isinstance(data, list):
                for r in data:
                    if isinstance(r, dict):
                        yield (report_name, r, {})
            return

        if isinstance(payload, dict) and "records" not in payload:
            for k, v in payload.items():
                if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], dict)):
                    for r in v:
                        if isinstance(r, dict):
                            yield (to_str(k), r, {})
            return

        if isinstance(payload, list) and (len(payload) == 0 or isinstance(payload[0], dict)):
            for r in payload:
                if isinstance(r, dict):
                    yield ("", r, {})
            return

        if isinstance(payload, list):
            for item in payload:
                yield from iter_reports_and_rows(item)
            return

        return

    # -------------------- main flow --------------------
    report_count = 0
    row_count = 0
    chunk_count = 0

    buffer_texts: List[str] = []
    buffer_metas: List[Dict[str, Any]] = []

    def flush():
        nonlocal buffer_texts, buffer_metas
        if buffer_texts:
            print(f"[INFO] Flushing {len(buffer_texts)} chunks to embed_fn...")
            embed_fn(buffer_texts, buffer_metas)
            buffer_texts, buffer_metas = [], []
            print("[INFO] Flush completed.")

    seen_report_names = set()

    for report_name, row, ctx in iter_reports_and_rows(zoho_data):
        if report_name not in seen_report_names:
            seen_report_names.add(report_name)
            report_count += 1
            print(f"[INFO] Processing report: '{report_name}'")

        row_count += 1
        print(f"[INFO] Processing row {row_count} (report: '{report_name}')")

        row_pk = stable_row_id(row)
        owner = ctx.get("owner_name", "")
        app = ctx.get("app_link_name", "")

        owner_s = slug(owner)
        app_s = slug(app)
        report_s = slug(report_name) if report_name else "unknown-report"

        preview = to_str(row.get("Full_Name") or row.get("Name") or row.get("Title") or row_pk)[:120]

        text = row_to_text(row)
        chunks = split_text(text, chunk_size, chunk_overlap)
        chunk_total = len(chunks)
        chunk_count += chunk_total
        print(f"[INFO] Row {row_count} split into {chunk_total} chunk(s)")

        group_id = f"{id_prefix}:{owner_s}:{app_s}:{report_s}:{row_pk}"
        row_hash = content_hash(row)

        for i, ch in enumerate(chunks, 1):
            vector_id = f"{group_id}:attributes:{i}"
            meta = {
                "source": "zoho",
                "namespace": namespace,
                "owner_name": owner,
                "app_link_name": app,
                "report": report_name,
                "row_pk": row_pk,
                "group_id": group_id,
                "field": "attributes",
                "chunk_no": i,
                "chunk_total": chunk_total,
                "text_content": ch,
                "preview": preview,
                "content_hash": row_hash,
                "vector_id": vector_id,
            }
            buffer_texts.append(ch)
            buffer_metas.append(meta)

            if len(buffer_texts) >= embed_batch:
                flush()

    flush()
    print(f"[INFO] Finished embedding: {report_count} reports, {row_count} rows, {chunk_count} chunks processed.")
    COMMON.save_json_data({"reports": report_count, "rows": row_count, "chunks": chunk_count, "namespace" : namespace})
    return {"reports": report_count, "rows": row_count, "chunks": chunk_count, "namespace" : namespace, "job_id": job_id}
