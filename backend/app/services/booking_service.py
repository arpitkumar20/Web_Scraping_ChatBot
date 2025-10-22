# # # app/services/booking_service.py
# # import os
# # import json
# # import datetime as dt
# # from typing import Dict, List, Tuple, Optional

# # # Reuse your embedding pipeline helpers
# # from app.services.embeddings_store_v2 import (
# #     create_documents, split_documents, generate_embeddings,
# #     init_pinecone_index, upsert_vectors
# # )
# # from app.core.config import (
# #     EMBEDDING_MODEL, OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX, PINECONE_ENV
# # )

# # # ------------------ storage paths ------------------
# # BOOKINGS_PATH = os.path.join("web_info", "bookings.json")
# # WEB_INFO_PATH = os.path.join("web_info", "web_info.json")

# # # ------------------ basic time helpers ------------------
# # TIME_FMT = "%H:%M"

# # def _to_time(t: str) -> dt.time:
# #     """Parse 'HH:MM' to time(). Raises ValueError if invalid."""
# #     return dt.datetime.strptime(t, TIME_FMT).time()

# # def _fmt_ampm(hhmm_24: str) -> str:
# #     """Format 'HH:MM' into 'h:mm AM/PM' without leading zero."""
# #     t = dt.datetime.strptime(hhmm_24, "%H:%M")
# #     out = t.strftime("%I:%M %p")
# #     return out.lstrip("0").replace(" 0", " ")

# # # ------------------ file persistence ------------------
# # def _load_bookings() -> Dict:
# #     if not os.path.exists(BOOKINGS_PATH):
# #         return {"bookings": []}
# #     with open(BOOKINGS_PATH, "r", encoding="utf-8") as f:
# #         try:
# #             return json.load(f)
# #         except Exception:
# #             return {"bookings": []}

# # def _save_bookings(data: Dict) -> None:
# #     os.makedirs(os.path.dirname(BOOKINGS_PATH), exist_ok=True)
# #     with open(BOOKINGS_PATH, "w", encoding="utf-8") as f:
# #         json.dump(data, f, indent=2, ensure_ascii=False)

# # def get_current_namespace() -> str:
# #     """
# #     Read namespace from web_info/web_info.json if present; else 'default'.
# #     (You can ignore this and pass namespace explicitly from webhook if you prefer.)
# #     """
# #     try:
# #         if os.path.exists(WEB_INFO_PATH):
# #             with open(WEB_INFO_PATH, "r", encoding="utf-8") as f:
# #                 j = json.load(f)
# #                 ns = j.get("namespace") or j.get("name") or j.get("company_name")
# #                 if ns:
# #                     return str(ns)
# #     except Exception:
# #         pass
# #     return "default"

# # # ------------------ booked & available computation ------------------
# # def list_booked_slots(doctor: str, date: str) -> List[Tuple[str, str]]:
# #     """
# #     Return list of (start,end) strings in 'HH:MM' for a doctor on a date from bookings.json.
# #     """
# #     data = _load_bookings()
# #     out = []
# #     for b in data.get("bookings", []):
# #         if b.get("doctor") == doctor and b.get("date") == date:
# #             out.append((b.get("start"), b.get("end")))
# #     out.sort(key=lambda x: x[0])
# #     return out

# # def generate_available_30min_slots(
# #     window_start: str,
# #     window_end: str,
# #     booked: List[Tuple[str, str]]
# # ) -> List[Tuple[str, str]]:
# #     """
# #     Given a working window ('HH:MM') and booked intervals ('HH:MM'),
# #     produce all free 30-min slots within window.
# #     """
# #     sw, ew = _to_time(window_start), _to_time(window_end)
# #     today = dt.date.today()
# #     cur = dt.datetime.combine(today, sw)
# #     end = dt.datetime.combine(today, ew)
# #     delta = dt.timedelta(minutes=30)
# #     slots = []
# #     while cur + delta <= end:
# #         s = cur.time().strftime(TIME_FMT)
# #         e = (cur + delta).time().strftime(TIME_FMT)
# #         slots.append((s, e))
# #         cur += delta

# #     def overlaps(a: Tuple[str,str], b: Tuple[str,str]) -> bool:
# #         a1, a2 = a; b1, b2 = b
# #         return not (a2 <= b1 or b2 <= a1)

# #     free = [s for s in slots if not any(overlaps(s, b) for b in booked)]
# #     return free

# # # ------------------ validation ------------------
# # def _end_from_start_and_duration(start_hhmm: str, duration_min: int) -> str:
# #     base = dt.datetime.strptime(start_hhmm, TIME_FMT)
# #     end = base + dt.timedelta(minutes=duration_min)
# #     return end.strftime(TIME_FMT)

# # def _is_within_window(start_hhmm: str, end_hhmm: str, window_start: str, window_end: str) -> bool:
# #     s = _to_time(start_hhmm)
# #     e = _to_time(end_hhmm)
# #     ws = _to_time(window_start)
# #     we = _to_time(window_end)
# #     return (ws <= s) and (e <= we) and (s < e)

# # def _conflicts_with_booked(start_hhmm: str, end_hhmm: str, booked: List[Tuple[str, str]]) -> bool:
# #     """True if requested interval overlaps any booked interval."""
# #     for bstart, bend in booked:
# #         # overlap if NOT (requested ends before bstart OR bend ends before requested start)
# #         if not (end_hhmm <= bstart or bend <= start_hhmm):
# #             return True
# #     return False

# # # --- NEW: ID & normalization helpers for Pinecone ---
# # def _norm_hhmm(s: str) -> str:
# #     s = (s or "").strip()
# #     h, m = s.split(":")
# #     return f"{int(h):02d}:{int(m):02d}"

# # def _sanitize_id_piece(s: str) -> str:
# #     return "".join(ch for ch in s if ch.isalnum() or ch in ("-", "_")).strip().replace(" ", "_")

# # def _make_booking_id(doctor: str, date: str, start: str) -> str:
# #     # deterministic ID so duplicate upserts overwrite the same record
# #     return f"booking::{_sanitize_id_piece(doctor)}::{date}::{_norm_hhmm(start)}"

# # # --- NEW: Query Pinecone for existing bookings by metadata filter ---
# # def _list_booked_slots_from_pinecone(namespace: str, doctor: str, date: str) -> List[Tuple[str, str]]:
# #     """
# #     Returns [(start,end), ...] for a doctor+date by reading Pinecone metadata.
# #     Uses an embedding-based query + metadata filter (source=booking_record, doctor, date).
# #     """
# #     from app.services.embeddings_store_v2 import generate_embeddings, init_pinecone_index

# #     # build a small query vector (same embed model ensures matching dims)
# #     probe_text = f"BOOKING_RECORD {doctor} {date}"
# #     vectors = generate_embeddings([probe_text], EMBEDDING_MODEL, OPENAI_API_KEY)
# #     if not vectors:
# #         return []

# #     index = init_pinecone_index(
# #         api_key=PINECONE_API_KEY,
# #         index_name=PINECONE_INDEX,
# #         dimension=len(vectors[0]),
# #         cloud="aws",
# #         region=PINECONE_ENV
# #     )

# #     # Filter by metadata so we only read booking records for doctor+date
# #     flt = {
# #         "source": {"$eq": "booking_record"},
# #         "doctor": {"$eq": doctor},
# #         "date": {"$eq": date},
# #     }

# #     try:
# #         # top_k is generous; include_metadata so we can read start/end
# #         resp = index.query(
# #             vector=vectors[0],
# #             top_k=200,
# #             include_metadata=True,
# #             namespace=namespace,
# #             filter=flt
# #         )
# #     except Exception as e:
# #         print("Pinecone query failed:", e)
# #         return []

# #     out: List[Tuple[str, str]] = []
# #     for match in (resp.get("matches") or []):
# #         md = match.get("metadata") or {}
# #         s, e = md.get("start"), md.get("end")
# #         if s and e:
# #             try:
# #                 out.append((_norm_hhmm(s), _norm_hhmm(e)))
# #             except Exception:
# #                 pass
# #     # dedupe + sort
# #     out = sorted(list({tuple(x) for x in out}), key=lambda t: t[0])
# #     return out

# # # ------------------ embedding persistence ------------------
# # def _embed_booking_record(namespace: str, entry: Dict) -> None:
# #     """
# #     Upserts a single vector with deterministic ID and rich metadata so we can later
# #     filter by (doctor,date) and read start/end directly from metadata.
# #     """
# #     note = (
# #         f"BOOKING_RECORD | Doctor: {entry['doctor']} | Date: {entry['date']} "
# #         f"| Slot: {entry['start']}-{entry['end']} | BookedBy: {entry.get('booked_by','unknown')}"
# #     )
# #     try:
# #         docs = create_documents([note], [{"source": "booking_record"}])
# #         chunks = split_documents(docs, chunk_size=1000, chunk_overlap=0)
# #         vectors = generate_embeddings(chunks, EMBEDDING_MODEL, OPENAI_API_KEY)
# #         if not vectors:
# #             return

# #         index = init_pinecone_index(
# #             api_key=PINECONE_API_KEY,
# #             index_name=PINECONE_INDEX,
# #             dimension=len(vectors[0]),
# #             cloud="aws",
# #             region=PINECONE_ENV
# #         )

# #         # IMPORTANT: include start/end in metadata, and use deterministic ID
# #         meta = {
# #             "source": "booking_record",
# #             "doctor": entry["doctor"],
# #             "date": entry["date"],
# #             "start": _norm_hhmm(entry["start"]),
# #             "end": _norm_hhmm(entry["end"]),
# #             "booked_by": entry.get("booked_by", "unknown"),
# #             "created_at": entry.get("created_at"),
# #         }
# #         vec_id = _make_booking_id(entry["doctor"], entry["date"], entry["start"])

# #         # Upsert exactly one vector
# #         index.upsert(
# #             vectors=[(vec_id, vectors[0], meta)],
# #             namespace=namespace
# #         )

# #     except Exception as e:
# #         print("Error embedding booking record:", e)

# # # ------------------ main API you call from the webhook ------------------
# # def validate_and_book(
# #     *,
# #     namespace: str,
# #     doctor: str,
# #     date: str,                 # 'YYYY-MM-DD' (webhook should validate this)
# #     start: str,                # 'HH:MM' 24h (webhook ensures this format)
# #     duration_min: int,         # typically 30
# #     window_start: str,         # 'HH:MM' 24h (passed from webhook/context)
# #     window_end: str,           # 'HH:MM' 24h
# #     booked_by: Optional[str] = None
# # ) -> Dict:
# #     """
# #     Pure, deterministic booking:
# #     - NO regex, NO parsing; assumes the webhook already extracted all fields.
# #     - Validates against the passed working window and existing bookings.
# #     - Persists to bookings.json and embeds a booking record in Pinecone.

# #     Returns:
# #       {
# #         "ok": bool,
# #         "message": str,              # human message you can send to user
# #         "conflict": bool,            # true if conflicts
# #         "slot": (start, end),
# #         "window": (window_start, window_end),
# #         "booked": [ (s,e), ... ]     # current day's booked after successful booking (or before if rejected)
# #       }
# #     """
# #     # Normalize inputs to HH:MM to avoid string compare issues
# #     try:
# #         start = _norm_hhmm(start)
# #         window_start = _norm_hhmm(window_start)
# #         window_end = _norm_hhmm(window_end)
# #     except Exception:
# #         return {"ok": False, "message": "Invalid time format. Use HH:MM (24h).", "conflict": False}

# #     # Compute end time
# #     try:
# #         end = _end_from_start_and_duration(start, duration_min)
# #     except Exception:
# #         return {"ok": False, "message": "Invalid time format for start or duration.", "conflict": False}

# #     # Load current bookings from Pinecone (source of truth), and also local for safety
# #     pc_booked = _list_booked_slots_from_pinecone(namespace, doctor, date)
# #     local_booked = list_booked_slots(doctor, date)

# #     # Merge unique; Pinecone-first
# #     seen = set()
# #     todays_booked: List[Tuple[str, str]] = []
# #     for s, e in pc_booked + local_booked:
# #         key = ( _norm_hhmm(s), _norm_hhmm(e) )
# #         if key not in seen:
# #             seen.add(key)
# #             todays_booked.append(key)

# #     # Validate within working window
# #     if not _is_within_window(start, end, window_start, window_end):
# #         msg = (
# #             f"Requested {start}-{end} is outside working hours "
# #             f"{window_start}-{window_end}."
# #         )
# #         return {
# #             "ok": False,
# #             "message": msg,
# #             "conflict": False,
# #             "slot": (start, end),
# #             "window": (window_start, window_end),
# #             "booked": todays_booked
# #         }

# #     # Validate not conflicting with existing bookings
# #     if _conflicts_with_booked(start, end, todays_booked):
# #         # Suggest free slots
# #         free_now = generate_available_30min_slots(window_start, window_end, todays_booked)
# #         suggestions = ", ".join([f"{_fmt_ampm(a)}–{_fmt_ampm(b)}" for a, b in free_now[:16]]) or "none"
# #         msg = (
# #             "That slot is already taken. Here are some free 30-min options:\n"
# #             f"{suggestions}"
# #         )
# #         return {
# #             "ok": False,
# #             "message": msg,
# #             "conflict": True,
# #             "slot": (start, end),
# #             "window": (window_start, window_end),
# #             "booked": todays_booked
# #         }

# #     # Persist & embed
# #     entry = {
# #         "doctor": doctor,
# #         "date": date,
# #         "start": start,
# #         "end": end,
# #         "booked_by": booked_by or "unknown",
# #         "created_at": dt.datetime.utcnow().isoformat() + "Z"
# #     }
# #     data = _load_bookings()
# #     data.setdefault("bookings", []).append(entry)
# #     _save_bookings(data)
# #     _embed_booking_record(namespace, entry)

# #     # Prepare success message
# #     message = (
# #         f"✅ Booking confirmed: {doctor} on {date} "
# #         f"{_fmt_ampm(start)}–{_fmt_ampm(end)}"
# #     )
# #     # Recompute booked for return clarity
# #     todays_booked = list_booked_slots(doctor, date)

# #     return {
# #         "ok": True,
# #         "message": message,
# #         "conflict": False,
# #         "slot": (start, end),
# #         "window": (window_start, window_end),
# #         "booked": todays_booked
# #     }


# # app/services/booking_service.py
# import os
# import json
# import re
# import datetime as dt
# from typing import Any, Callable, Dict, List, Optional, Tuple

# # Reuse your embedding pipeline helpers
# from app.services.embeddings_store_v2 import (
#     create_documents, split_documents, generate_embeddings,
#     init_pinecone_index, upsert_vectors
# )
# from app.core.config import (
#     EMBEDDING_MODEL, OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX, PINECONE_ENV
# )

# # ------------------ storage paths ------------------
# BOOKINGS_PATH = os.path.join("web_info", "bookings.json")
# WEB_INFO_PATH = os.path.join("web_info", "web_info.json")

# # ------------------ booking-specific regex/helpers ------------------
# TIME_FMT = "%H:%M"
# BOOKING_CONFIRMATION_RE = re.compile(
#     r"^\s*CONFIRM_BOOKING:\s*doctor=(?P<doctor>.+?);\s*date=(?P<date>\d{4}-\d{2}-\d{2});\s*start=(?P<start>\d{2}:\d{2});\s*duration=(?P<duration>\d+)\s*$",
#     re.IGNORECASE
# )

# AVAIL_KEYWORDS = (
#     "available slot", "availability", "available", "book",
#     "schedule", "timing", "timings", "appointment"
# )

# DOCTOR_NAME_RE = re.compile(r"(Dr\.?\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)")
# DATE_RE_ISO = re.compile(r"(\d{4}-\d{2}-\d{2})")

# DAY_ABBR = r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun)"
# WINDOW_RE = re.compile(
#     rf"(?P<days>{DAY_ABBR}(?:\s*[–-]\s*{DAY_ABBR})?)\s+"
#     r"(?P<start>\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))\s*[-–]\s*"
#     r"(?P<end>\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))"
# )

# # ------------------ file persistence ------------------
# def _load_bookings() -> Dict:
#     if not os.path.exists(BOOKINGS_PATH):
#         return {"bookings": []}
#     with open(BOOKINGS_PATH, "r", encoding="utf-8") as f:
#         try:
#             return json.load(f)
#         except Exception:
#             return {"bookings": []}

# def _save_bookings(data: Dict) -> None:
#     os.makedirs(os.path.dirname(BOOKINGS_PATH), exist_ok=True)
#     with open(BOOKINGS_PATH, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2, ensure_ascii=False)

# def get_current_namespace() -> str:
#     """
#     Read namespace from web_info/web_info.json if present; else 'default'.
#     """
#     try:
#         if os.path.exists(WEB_INFO_PATH):
#             with open(WEB_INFO_PATH, "r", encoding="utf-8") as f:
#                 j = json.load(f)
#                 ns = j.get("namespace") or j.get("name") or j.get("company_name")
#                 if ns:
#                     return str(ns)
#     except Exception:
#         pass
#     return "default"

# # ------------------ format/parse helpers ------------------
# def _to_24h(hhmm_ampm: str) -> str:
#     s = hhmm_ampm.strip().upper().replace(" ", "")
#     fmt = "%I%p" if ":" not in s else "%I:%M%p"
#     return dt.datetime.strptime(s, fmt).strftime("%H:%M")

# def _fmt_ampm(hhmm_24: str) -> str:
#     t = dt.datetime.strptime(hhmm_24, "%H:%M")
#     out = t.strftime("%I:%M %p")
#     return out.lstrip("0").replace(" 0", " ")

# def _stringify_context(ctx: Any) -> str:
#     if ctx is None:
#         return ""
#     if isinstance(ctx, (list, tuple)):
#         return " ".join(map(str, ctx))
#     if isinstance(ctx, dict):
#         return " ".join(map(str, ctx.values()))
#     return str(ctx)

# def _extract_window_from_text(text: str, doctor_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
#     """
#     Parse 'Mon-Sat 8 AM - 4 PM' near the doctor's line, else anywhere in text.
#     Returns (days_str, start_24, end_24) or (None, None, None).
#     """
#     lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
#     for ln in lines:
#         if doctor_name.lower() in ln.lower():
#             m = WINDOW_RE.search(ln)
#             if m:
#                 days = m.group("days").replace("–", "-").replace(" ", "")
#                 return days, _to_24h(m.group("start")), _to_24h(m.group("end"))
#     m = WINDOW_RE.search(text)
#     if m:
#         days = m.group("days").replace("–", "-").replace(" ", "")
#         return days, _to_24h(m.group("start")), _to_24h(m.group("end"))
#     return None, None, None

# # ------------------ booking computations ------------------
# def list_booked_slots(doctor: str, date: str) -> List[Tuple[str, str]]:
#     data = _load_bookings()
#     out = []
#     for b in data.get("bookings", []):
#         if b.get("doctor") == doctor and b.get("date") == date:
#             out.append((b.get("start"), b.get("end")))
#     out.sort(key=lambda x: x[0])
#     return out

# def _to_time(t: str) -> dt.time:
#     return dt.datetime.strptime(t, TIME_FMT).time()

# def generate_available_30min_slots(start_window: str, end_window: str,
#                                    booked: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
#     sw, ew = _to_time(start_window), _to_time(end_window)
#     cur = dt.datetime.combine(dt.date.today(), sw)
#     end = dt.datetime.combine(dt.date.today(), ew)
#     delta = dt.timedelta(minutes=30)
#     slots = []
#     while cur + delta <= end:
#         s = cur.time().strftime(TIME_FMT)
#         e = (cur + delta).time().strftime(TIME_FMT)
#         slots.append((s, e))
#         cur += delta

#     def overlaps(a: Tuple[str,str], b: Tuple[str,str]) -> bool:
#         a1, a2 = a; b1, b2 = b
#         return not (a2 <= b1 or b2 <= a1)

#     free = [s for s in slots if not any(overlaps(s, b) for b in booked)]
#     return free

# def _format_slots(slots: List[Tuple[str, str]], limit: int = 8) -> str:
#     if not slots:
#         return "none"
#     return ", ".join([f"{_fmt_ampm(a)}–{_fmt_ampm(b)}" for a, b in slots[:limit]])

# # ------------------ embedding persistence ------------------
# def _embed_booking_record(namespace: str, entry: Dict) -> None:
#     note = (
#         f"BOOKING_RECORD | Doctor: {entry['doctor']} | Date: {entry['date']} "
#         f"| Slot: {entry['start']}-{entry['end']} | BookedBy: {entry.get('booked_by','unknown')}"
#     )
#     try:
#         docs = create_documents([note], [{"source": "booking_record"}])
#         chunks = split_documents(docs, chunk_size=1000, chunk_overlap=0)
#         vectors = generate_embeddings(chunks, EMBEDDING_MODEL, OPENAI_API_KEY)
#         if not vectors:
#             return
#         index = init_pinecone_index(
#             api_key=PINECONE_API_KEY,
#             index_name=PINECONE_INDEX,
#             dimension=len(vectors[0]),
#             cloud="aws",
#             region=PINECONE_ENV
#         )
#         metas = [{"source": "booking_record", "doctor": entry["doctor"], "date": entry["date"]} for _ in chunks]
#         upsert_vectors(index, chunks, vectors, namespace, metas)
#     except Exception as e:
#         print("Error embedding booking record:", e)

# def persist_booking_and_embed(namespace: str, payload: Dict) -> None:
#     data = _load_bookings()
#     entry = {
#         "doctor": payload["doctor"],
#         "date": payload["date"],
#         "start": payload["start"],
#         "end": payload["end"],
#         "booked_by": payload.get("booked_by", "unknown"),
#         "created_at": dt.datetime.utcnow().isoformat() + "Z"
#     }
#     data.setdefault("bookings", []).append(entry)
#     _save_bookings(data)
#     _embed_booking_record(namespace, entry)

# # ------------------ intent & parsing ------------------
# def extract_doctor_name(text: str) -> Optional[str]:
#     if not text:
#         return None
#     m = DOCTOR_NAME_RE.search(text)
#     if m:
#         return m.group(0).strip()
#     return None

# def extract_date_yyyy_mm_dd(text: str) -> str:
#     if not text:
#         return dt.date.today().isoformat()
#     m = DATE_RE_ISO.search(text)
#     if m:
#         return m.group(1)
#     return dt.date.today().isoformat()

# def try_parse_confirmation(text: str) -> Optional[Dict]:
#     if not isinstance(text, str):
#         return None
#     m = BOOKING_CONFIRMATION_RE.match(text.strip())
#     if not m:
#         return None
#     d = m.groupdict()
#     d["duration"] = int(d["duration"])
#     start_dt = dt.datetime.strptime(d["start"], TIME_FMT)
#     end_dt = start_dt + dt.timedelta(minutes=d["duration"])
#     d["end"] = end_dt.strftime(TIME_FMT)
#     return d

# def is_availability_query(text: str) -> bool:
#     if not text:
#         return False
#     t = text.lower()
#     return any(k in t for k in AVAIL_KEYWORDS)

# # ------------------ context-based window query ------------------
# def get_doctor_window_from_context(
#     namespace: str,
#     doctor_name: str,
#     retriever_fn: Callable[[str], Any]
# ) -> Tuple[str, str, str]:
#     """
#     Ask retriever for text and parse window.
#     retriever_fn: a function you pass in from webhook, e.g.
#       lambda q: query_pinecone_index(q, namespace=namespace)
#     Returns (days_str, start_24, end_24) with fallback.
#     """
#     ctx = retriever_fn(f"{doctor_name} available days timing schedule hours")
#     text = _stringify_context(ctx)
#     days, start_24, end_24 = _extract_window_from_text(text, doctor_name)
#     if not (days and start_24 and end_24):
#         days, start_24, end_24 = "Mon-Sat", "08:00", "16:00"
#     return days, start_24, end_24

# # ------------------ public entry: single call from webhook ------------------
# def prepare_booking_response(
#     *,
#     namespace: str,
#     message_text: str,
#     booked_by: str,
#     retriever_fn: Callable[[str], Any]
# ) -> Dict[str, Any]:
#     """
#     Single entry for webhook to call.
#     - Detects confirmation or availability.
#     - If confirmation: persists + embeds, returns confirmation message.
#     - If availability: computes window from context, returns clean availability message.
#     - If neither: returns {"handled": False}.
#     Webhook keeps control of messageId/logging/sending.
#     """
#     message_text = (message_text or "").strip()

#     # 1) Confirmation path
#     parsed = try_parse_confirmation(message_text)
#     if parsed:
#         # Pull booked; validate requested slot is free against window
#         # (optional window validation: fetch from context)
#         days_str, w_start, w_end = get_doctor_window_from_context(namespace, parsed["doctor"], retriever_fn)
#         booked_now = list_booked_slots(parsed["doctor"], parsed["date"])
#         free_now = generate_available_30min_slots(w_start, w_end, booked_now)
#         requested = (parsed["start"], parsed["end"])

#         if requested not in free_now:
#             suggestion = _format_slots(free_now, limit=8)
#             msg = (
#                 "That slot isn't available. Here are some free 30-min options:\n"
#                 f"{suggestion}\n"
#                 "Please resend with your chosen start time.\n"
#                 "Format:\n"
#                 f"CONFIRM_BOOKING: doctor={parsed['doctor']}; date={parsed['date']}; start=HH:MM; duration=30"
#             )
#             return {
#                 "handled": True,
#                 "intent": "confirmation_rejected",
#                 "reply_text": msg,
#                 "meta": {
#                     "doctor": parsed["doctor"],
#                     "date": parsed["date"],
#                     "requested": requested,
#                     "window": (w_start, w_end),
#                     "booked": booked_now[:50],
#                 }
#             }

#         parsed["booked_by"] = booked_by
#         persist_booking_and_embed(namespace, parsed)
#         conf_msg = (
#             f"✅ Booking confirmed: {parsed['doctor']} on {parsed['date']} "
#             f"{_fmt_ampm(parsed['start'])}-{_fmt_ampm(parsed['end'])}"
#         )
#         return {
#             "handled": True,
#             "intent": "confirmation_accepted",
#             "reply_text": conf_msg,
#             "meta": {
#                 "doctor": parsed["doctor"],
#                 "date": parsed["date"],
#                 "slot": (parsed["start"], parsed["end"]),
#                 "window": (w_start, w_end)
#             }
#         }

#     # 2) Availability path
#     if is_availability_query(message_text):
#         doctor_asked = extract_doctor_name(message_text)
#         date_asked = extract_date_yyyy_mm_dd(message_text)

#         if not doctor_asked:
#             prompt = (
#                 "Please share the doctor's name (e.g., 'Dr. Sneha Singh') "
#                 "and the date as YYYY-MM-DD."
#             )
#             return {"handled": True, "intent": "need_doctor", "reply_text": prompt, "meta": {}}

#         days_str, work_start, work_end = get_doctor_window_from_context(namespace, doctor_asked, retriever_fn)
#         booked = list_booked_slots(doctor_asked, date_asked)
#         free_slots = generate_available_30min_slots(work_start, work_end, booked)

#         msg = (
#             f"{doctor_asked} is available {days_str} {_fmt_ampm(work_start)}–{_fmt_ampm(work_end)}.\n"
#             f"Booked on {date_asked}: {_format_slots(booked, limit=99)}\n"
#             f"Sample free 30-min slots: {_format_slots(free_slots, limit=8)}\n"
#             f"To book, reply exactly:\n"
#             f"CONFIRM_BOOKING: doctor={doctor_asked}; date={date_asked}; start=HH:MM; duration=30"
#         )
#         return {
#             "handled": True,
#             "intent": "availability",
#             "reply_text": msg,
#             "meta": {
#                 "doctor": doctor_asked,
#                 "date": date_asked,
#                 "window": (work_start, work_end),
#                 "booked_count": len(booked)
#             }
#         }

#     # 3) Not handled here -> let webhook do normal RAG/LLM
#     return {"handled": False}













import os
import json
import re
import datetime as dt
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.services.embeddings_store_v2 import (
    create_documents, split_documents, generate_embeddings,
    init_pinecone_index, upsert_vectors
)
from app.core.config import (
    EMBEDDING_MODEL, OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX, PINECONE_ENV
)

import pinecone

# ------------------ storage paths ------------------
BOOKINGS_PATH = os.path.join("web_info", "bookings.json")
WEB_INFO_PATH = os.path.join("web_info", "web_info.json")

# ------------------ regex ------------------
TIME_FMT = "%H:%M"

BOOKING_CONFIRMATION_RE = re.compile(
    r"^\s*CONFIRM_BOOKING:\s*doctor=(?P<doctor>.+?);\s*date=(?P<date>\d{4}-\d{2}-\d{2});\s*start=(?P<start>\d{2}:\d{2});\s*duration=(?P<duration>\d+)\s*$",
    re.IGNORECASE
)

AVAIL_KEYWORDS = (
    "slot", "availability", "available", "book",
    "schedule", "timing", "time", "appointment"
)

DOCTOR_NAME_RE = re.compile(r"(Dr\.?\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)")
DATE_RE_ISO = re.compile(r"(\d{4}-\d{2}-\d{2})")

DAY_ABBR = r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun)"
WINDOW_RE = re.compile(
    rf"(?P<days>{DAY_ABBR}(?:\s*[–-]\s*{DAY_ABBR})?)\s+"
    r"(?P<start>\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))\s*[-–]\s*"
    r"(?P<end>\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm))"
)

# ------------------ helpers ------------------
def _to_24h(hhmm_ampm: str) -> str:
    s = hhmm_ampm.strip().upper().replace(" ", "")
    fmt = "%I%p" if ":" not in s else "%I:%M%p"
    return dt.datetime.strptime(s, fmt).strftime("%H:%M")

def _fmt_ampm(hhmm_24: str) -> str:
    t = dt.datetime.strptime(hhmm_24, "%H:%M")
    out = t.strftime("%I:%M %p")
    return out.lstrip("0").replace(" 0", " ")

def _stringify_context(ctx: Any) -> str:
    if ctx is None:
        return ""
    if isinstance(ctx, (list, tuple)):
        return " ".join(map(str, ctx))
    if isinstance(ctx, dict):
        return " ".join(map(str, ctx.values()))
    return str(ctx)

def _to_time(t: str) -> dt.time:
    return dt.datetime.strptime(t, TIME_FMT).time()

# ------------------ file persistence ------------------
def _load_bookings() -> Dict:
    if not os.path.exists(BOOKINGS_PATH):
        return {"bookings": []}
    with open(BOOKINGS_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {"bookings": []}

def _save_bookings(data: Dict) -> None:
    os.makedirs(os.path.dirname(BOOKINGS_PATH), exist_ok=True)
    with open(BOOKINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_current_namespace() -> str:
    try:
        if os.path.exists(WEB_INFO_PATH):
            with open(WEB_INFO_PATH, "r", encoding="utf-8") as f:
                j = json.load(f)
                ns = j.get("namespace") or j.get("name") or j.get("company_name")
                if ns:
                    return str(ns)
    except Exception:
        pass
    return "default"

# ------------------ booking window extraction ------------------
def _extract_window_from_text(text: str, doctor_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        if doctor_name.lower() in ln.lower():
            m = WINDOW_RE.search(ln)
            if m:
                days = m.group("days").replace("–", "-").replace(" ", "")
                return days, _to_24h(m.group("start")), _to_24h(m.group("end"))
    m = WINDOW_RE.search(text)
    if m:
        days = m.group("days").replace("–", "-").replace(" ", "")
        return days, _to_24h(m.group("start")), _to_24h(m.group("end"))
    return None, None, None

def get_doctor_window_from_context(namespace: str, doctor_name: str, retriever_fn: Callable[[str], Any]) -> Tuple[str, str, str]:
    ctx = retriever_fn(f"{doctor_name} available days timing schedule hours")
    text = _stringify_context(ctx)
    days, start_24, end_24 = _extract_window_from_text(text, doctor_name)
    if not (days and start_24 and end_24):
        days, start_24, end_24 = "Mon-Sat", "08:00", "16:00"
    return days, start_24, end_24

# ------------------ Pinecone query helper ------------------
def query_pinecone_bookings(query_text: str, namespace: str = "default", top_k: int = 100) -> List[Dict]:
    """
    Query Pinecone index for booking records related to a doctor/date.
    Returns list of metadata dicts (start/end slots, doctor, date).
    """
    try:
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        index = pinecone.Index(PINECONE_INDEX)

        # Create embedding for the query
        query_vector = generate_embeddings([query_text], EMBEDDING_MODEL, OPENAI_API_KEY)[0]

        res = index.query(vector=query_vector, top_k=top_k, include_metadata=True, namespace=namespace)
        bookings = []
        for match in res.matches:
            meta = match.metadata
            if meta.get("source") == "booking_record":
                bookings.append(meta)
        return bookings
    except Exception as e:
        print("Error querying Pinecone:", e)
        return []

# ------------------ slot calculations ------------------
def list_booked_slots(doctor: str, date: str, namespace: str) -> List[Tuple[str, str]]:
    """
    Fetch all booked slots for a doctor on a date using local JSON + Pinecone.
    """
    booked_slots = []

    # Local JSON fallback
    local_bookings = _load_bookings()
    booked_slots.extend([
        (b["start"], b["end"])
        for b in local_bookings.get("bookings", [])
        if b.get("doctor") == doctor and b.get("date") == date
    ])

    # Pinecone bookings
    pinecone_bookings = query_pinecone_bookings(f"Doctor {doctor} booked on {date}", namespace)
    for meta in pinecone_bookings:
        booked_slots.append((meta.get("start"), meta.get("end")))

    # Remove duplicates and sort
    booked_slots = list({(s, e) for s, e in booked_slots})
    booked_slots.sort(key=lambda x: x[0])
    return booked_slots

def generate_available_30min_slots(start_window: str, end_window: str, booked: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    sw, ew = _to_time(start_window), _to_time(end_window)
    cur = dt.datetime.combine(dt.date.today(), sw)
    end = dt.datetime.combine(dt.date.today(), ew)
    delta = dt.timedelta(minutes=30)
    slots = []
    while cur + delta <= end:
        s = cur.time().strftime(TIME_FMT)
        e = (cur + delta).time().strftime(TIME_FMT)
        slots.append((s, e))
        cur += delta

    def overlaps(a: Tuple[str,str], b: Tuple[str,str]) -> bool:
        a1, a2 = a; b1, b2 = b
        return not (a2 <= b1 or b2 <= a1)

    free = [s for s in slots if not any(overlaps(s, b) for b in booked)]
    return free

def _format_slots(slots: List[Tuple[str, str]], limit: int = 8) -> str:
    if not slots:
        return "none"
    return ", ".join([f"{_fmt_ampm(a)}–{_fmt_ampm(b)}" for a, b in slots[:limit]])

# ------------------ embedding ------------------
def _embed_booking_record(namespace: str, entry: Dict) -> None:
    note = (
        f"BOOKING_RECORD | Doctor: {entry['doctor']} | Date: {entry['date']} "
        f"| Slot: {entry['start']}-{entry['end']} | BookedBy: {entry.get('booked_by','unknown')}"
    )
    try:
        docs = create_documents([note], [{"source": "booking_record"}])
        chunks = split_documents(docs, chunk_size=1000, chunk_overlap=0)
        vectors = generate_embeddings(chunks, EMBEDDING_MODEL, OPENAI_API_KEY)
        if not vectors:
            return
        index = init_pinecone_index(
            api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX,
            dimension=len(vectors[0]),
            cloud="aws",
            region=PINECONE_ENV
        )
        metas = [{"source": "booking_record", "doctor": entry["doctor"], "date": entry["date"], "start": entry["start"], "end": entry["end"]} for _ in chunks]
        upsert_vectors(index, chunks, vectors, namespace, metas)
    except Exception as e:
        print("Error embedding booking record:", e)

# ------------------ thread-safe booking ------------------
booking_lock = threading.Lock()

def persist_booking_and_embed_safe(namespace: str, payload: Dict) -> bool:
    with booking_lock:
        booked_now = list_booked_slots(payload["doctor"], payload["date"], namespace)
        requested = (payload["start"], payload["end"])

        def overlaps(a: Tuple[str,str], b: Tuple[str,str]) -> bool:
            a1, a2 = a; b1, b2 = b
            return not (a2 <= b1 or b2 <= a1)

        if any(overlaps(requested, b) for b in booked_now):
            return False

        # Save to JSON
        data = _load_bookings()
        entry = {
            "doctor": payload["doctor"],
            "date": payload["date"],
            "start": payload["start"],
            "end": payload["end"],
            "booked_by": payload.get("booked_by", "unknown"),
            "created_at": dt.datetime.utcnow().isoformat() + "Z"
        }
        data.setdefault("bookings", []).append(entry)
        _save_bookings(data)

        # Embed asynchronously
        threading.Thread(target=_embed_booking_record, args=(namespace, entry), daemon=True).start()
        return True

# ------------------ parsing ------------------
def extract_doctor_name(text: str) -> Optional[str]:
    if not text:
        return None
    m = DOCTOR_NAME_RE.search(text)
    if m:
        return m.group(0).strip()
    return None

def extract_date_yyyy_mm_dd(text: str) -> str:
    if not text:
        return dt.date.today().isoformat()
    m = DATE_RE_ISO.search(text)
    if m:
        return m.group(1)
    return dt.date.today().isoformat()

def try_parse_confirmation(text: str) -> Optional[Dict]:
    if not isinstance(text, str):
        return None
    m = BOOKING_CONFIRMATION_RE.match(text.strip())
    if not m:
        return None
    d = m.groupdict()
    d["duration"] = int(d["duration"])
    start_dt = dt.datetime.strptime(d["start"], TIME_FMT)
    end_dt = start_dt + dt.timedelta(minutes=d["duration"])
    d["end"] = end_dt.strftime(TIME_FMT)
    return d

def is_availability_query(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(k in t for k in AVAIL_KEYWORDS)

# ------------------ webhook-ready booking response ------------------
def prepare_booking_response(
    *,
    namespace: str,
    message_text: str,
    booked_by: str,
    retriever_fn: Callable[[str], Any]
) -> Dict[str, Any]:

    message_text = (message_text or "").strip()

    # 1) Confirmation
    parsed = try_parse_confirmation(message_text)
    if parsed:
        days_str, w_start, w_end = get_doctor_window_from_context(namespace, parsed["doctor"], retriever_fn)
        requested = (parsed["start"], parsed["end"])

        parsed["booked_by"] = booked_by
        success = persist_booking_and_embed_safe(namespace, parsed)
        if not success:
            booked_now = list_booked_slots(parsed["doctor"], parsed["date"], namespace)
            free_now = generate_available_30min_slots(w_start, w_end, booked_now)
            suggestion = _format_slots(free_now, limit=8)
            msg = (
                "⚠️ That slot is already booked. Here are some free 30-min options:\n"
                f"{suggestion}\n"
                "Please resend with your chosen start time.\n"
                "Format:\n"
                f"CONFIRM_BOOKING: doctor={parsed['doctor']}; date={parsed['date']}; start=HH:MM; duration=30"
            )
            return {
                "handled": True,
                "intent": "confirmation_rejected",
                "reply_text": msg,
                "meta": {"doctor": parsed["doctor"], "date": parsed["date"], "requested": requested}
            }

        conf_msg = (
            f"✅ Booking confirmed: {parsed['doctor']} on {parsed['date']} "
            f"{_fmt_ampm(parsed['start'])}-{_fmt_ampm(parsed['end'])}"
        )
        return {
            "handled": True,
            "intent": "confirmation_accepted",
            "reply_text": conf_msg,
            "meta": {"doctor": parsed["doctor"], "date": parsed["date"], "slot": requested}
        }

    # 2) Availability
    if is_availability_query(message_text):
        doctor_asked = extract_doctor_name(message_text)
        date_asked = extract_date_yyyy_mm_dd(message_text)

        if not doctor_asked:
            prompt = (
                "Please share the doctor's name (e.g., 'Dr. Sneha Singh') "
                "and the date as YYYY-MM-DD."
            )
            return {"handled": True, "intent": "need_doctor", "reply_text": prompt, "meta": {}}

        days_str, work_start, work_end = get_doctor_window_from_context(namespace, doctor_asked, retriever_fn)
        booked = list_booked_slots(doctor_asked, date_asked, namespace)
        free_slots = generate_available_30min_slots(work_start, work_end, booked)

        msg = (
            f"{doctor_asked} is available {days_str} {_fmt_ampm(work_start)}–{_fmt_ampm(work_end)}.\n"
            f"Booked on {date_asked}: {_format_slots(booked, limit=99)}\n"
            f"Sample free 30-min slots: {_format_slots(free_slots, limit=8)}\n"
            f"To book, reply exactly:\n"
            f"CONFIRM_BOOKING: doctor={doctor_asked}; date={date_asked}; start=HH:MM; duration=30"
        )
        return {
            "handled": True,
            "intent": "availability",
            "reply_text": msg,
            "meta": {
                "doctor": doctor_asked,
                "date": date_asked,
                "window": (work_start, work_end),
                "booked_count": len(booked)
            }
        }

    # 3) Not handled
    return {"handled": False}
