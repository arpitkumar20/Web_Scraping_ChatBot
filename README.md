# Web_Scraping_ChatBot

## Project Structure

```
WEB_SCRAPING_CHATBOT/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── database_scrap.py
│   │   │   └── (other API files you may have)
│   │   ├── __init__.py
│   │   ├── wsgi.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── embedding_utils.py
│   │   │   └── logging.py
│   │   ├── helper/
│   │   │   └── utils.py
│   │   ├── models/
│   │   │   ├── mysql_db.py
│   │   │   ├── postgresql_db.py
│   │   │   └── zoho_connectors.py
│   │   ├── prompts/
│   │   │   ├── namespace_prompt.py
│   │   │   └── qa_prompt.py
│   │   ├── services/
│   │   │   └── (your services files)
│   │   ├── web_scraping/
│   │   │   ├── scraper_function.py
│   │   │   └── web_scraper.py
│   │   └── requirements.txt
│   ├── logs/
│   └── venv/
│
├── web_info/
│   └── web_info.json
├── .env
├── .gitignore
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## Notes on Modifications
- `core/` contains `config.py`, `embedding_utils.py`, and `logging.py`.
- `helper/` contains `utils.py`.
- `models/` includes `mysql_db.py`, `postgresql_db.py`, and `zoho_connectors.py`.
- `prompts/` includes `namespace_prompt.py` and `qa_prompt.py`.
- Removed extra `auth` and `scripts` folders.
- `logs/` exists at the root of `backend`.
- `venv/` included under `backend`.

## Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/arpitkumar20/Web_Scraping_ChatBot.git
```

2. Create a virtual environment (first time only):
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
source venv/bin/activate
```

4. Upgrade pip:
```bash
python -m pip install --upgrade pip
```

5. Install dependencies:
```bash
pip install -r requirements.txt
```

6. Run REST (Flask) Service:
```bash
python -m app.service
```

## Conversational Intelligence (Q&A, RAG)

- **Backend services:**
  - `app/core/embedding_utils.py` → Handles embeddings and vector-related operations.
  - `app/prompts/qa_prompt.py` → Prompt templates for question-answering.
  - `app/prompts/namespace_prompt.py` → Prompt templates for context or namespace handling.

- **API routes:**
  - `app/api/database_scrap.py` → Endpoints that fetch data, scrape websites, and handle RAG workflows.

## Website Crawling & Indexing

- **Backend services:**
  - `app/web_scraping/scraper_function.py` → Core scraping functions for websites.
  - `app/web_scraping/web_scraper.py` → Higher-level scraping logic and orchestration.
  - `app/services/` → Services related to crawling, ingestion, and content processing.

## HMS & Booking

- **Backend services:**
  - Could be implemented inside `app/services/` (e.g., `hms_integration.py`, `hotel_integration.py`).

- **Data models:**
  - `app/models/` → Database models like `mysql_db.py` or `postgresql_db.py` for storing booking or HMS data.

## WhatsApp Integration

- **Backend services:**
  - Implement in `app/services/` (e.g., `whatsapp.py`) for sending/receiving messages.

- **API routes:**
  - Could create `app/api/routes_whatsapp.py` for message handling.

## Admin Panel

- **Backend:**
  - Admin-specific services can live under `app/services/admin/` (optional).
  - API routes like `routes_admin.py` in `app/api/`.

- **Frontend:**
  - Optional `frontend/` folder for dashboard UI.

## 360° Room Tours / Hotel Enhancements

- **Backend services:**
  - Could be implemented in `app/services/hotel_integration.py`.

- **Frontend:**
  - UI components for 360° tours would live in the frontend dashboard.

## Hospital Features

- **Data models:**
  - `app/models/` → Models for doctors, appointments, and hospital management.

- **Backend services:**
  - `app/services/hms_integration.py` → Handles hospital scheduling and appointments.

## DevOps / Infrastructure

- **Directories:**
  - `Dockerfile` → Container configuration.
  - `docker-compose.yml` → Multi-container setup for app + DB.
  - `.env` → Environment variables.
  - `backend/venv/` → Virtual environment (optional, usually ignored in git).
  - `logs/` → Logging outputs.

## Testing

- **Test suites:**
  - Add `tests/unit/`, `tests/integration/`, etc., at the root level alongside `backend/` for structured testing.

This mapping aligns exactly with your current folder structure, showing where each feature would live or be implemented. You can optionally create a visual folder-feature diagram for easier onboarding and documentation.
