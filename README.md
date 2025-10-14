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

# NGROK
```bash
Domain : aeronautic-showier-marquitta.ngrok-free.app
ngrok http 5004 --url aeronautic-showier-marquitta.ngrok-free.app
ngrok http --url=aeronautic-showier-marquitta.ngrok-free.app 5000
```
---

### TIKA SERVER
```bash
docker run -d -p 9998:9998 logicalspark/docker-tikaserver
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


## REDIS RUN COMMAND DOCKER:
```bash
docker run -d --name redisinsight -p 8001:8001 redislabs/redisinsight:latest
or,
docker run --name redis-local -p 6379:6379 -d redis redis-server --appendonly yes
```
## REDIS COMMAND:
```bash
npm install -g redis-commander
echo $PATH
npx redis-commander
```
<!-- ###
This mapping aligns exactly with your current folder structure, showing where each feature would live or be implemented. You can optionally create a visual folder-feature diagram for easier onboarding and documentation.


(venv) user@user-HP-Pavilion-Gaming-Laptop:~/Web_Scraping_ChatBot/backend$ pip install redis
Collecting redis
  Downloading redis-6.4.0-py3-none-any.whl.metadata (10 kB)
Requirement already satisfied: async-timeout>=4.0.3 in ./venv/lib/python3.10/site-packages (from redis) (4.0.3)
Downloading redis-6.4.0-py3-none-any.whl (279 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 279.8/279.8 kB 1.9 MB/s eta 0:00:00
Installing collected packages: redis
Successfully installed redis-6.4.0

[notice] A new release of pip is available: 23.3.1 -> 25.2
[notice] To update, run: pip install --upgrade pip
(venv) user@user-HP-Pavilion-Gaming-Laptop:~/Web_Scraping_ChatBot/backend$ redis-cli ping
Command 'redis-cli' not found, but can be installed with:
sudo apt install redis-tools
(venv) user@user-HP-Pavilion-Gaming-Laptop:~/Web_Scraping_ChatBot/backend$ sudo apt install redis-tools
[sudo] password for user: 
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
The following additional packages will be installed:
  libjemalloc2 liblua5.1-0 liblzf1 lua-bitop lua-cjson
Suggested packages:
  ruby-redis
The following NEW packages will be installed:
  libjemalloc2 liblua5.1-0 liblzf1 lua-bitop lua-cjson redis-tools
0 upgraded, 6 newly installed, 0 to remove and 48 not upgraded.
Need to get 1,227 kB of archives.
After this operation, 5,530 kB of additional disk space will be used.
Do you want to continue? [Y/n] Y
Get:1 http://in.archive.ubuntu.com/ubuntu jammy/universe amd64 libjemalloc2 amd64 5.2.1-4ubuntu1 [240 kB]
Get:2 http://in.archive.ubuntu.com/ubuntu jammy/universe amd64 liblua5.1-0 amd64 5.1.5-8.1build4 [99.9 kB]
Get:3 http://in.archive.ubuntu.com/ubuntu jammy/universe amd64 liblzf1 amd64 3.6-3 [7,444 B]
Get:4 http://in.archive.ubuntu.com/ubuntu jammy/universe amd64 lua-bitop amd64 1.0.2-5 [6,680 B]
Get:5 http://in.archive.ubuntu.com/ubuntu jammy/universe amd64 lua-cjson amd64 2.1.0+dfsg-2.1 [17.4 kB]
Get:6 http://in.archive.ubuntu.com/ubuntu jammy/universe amd64 redis-tools amd64 5:6.0.16-1ubuntu1 [856 kB]
Fetched 1,227 kB in 6s (220 kB/s)    
Selecting previously unselected package libjemalloc2:amd64.
(Reading database ... 286628 files and directories currently installed.)
Preparing to unpack .../0-libjemalloc2_5.2.1-4ubuntu1_amd64.deb ...
Unpacking libjemalloc2:amd64 (5.2.1-4ubuntu1) ...
Selecting previously unselected package liblua5.1-0:amd64.
Preparing to unpack .../1-liblua5.1-0_5.1.5-8.1build4_amd64.deb ...
Unpacking liblua5.1-0:amd64 (5.1.5-8.1build4) ...
Selecting previously unselected package liblzf1:amd64.
Preparing to unpack .../2-liblzf1_3.6-3_amd64.deb ...
Unpacking liblzf1:amd64 (3.6-3) ...
Selecting previously unselected package lua-bitop:amd64.
Preparing to unpack .../3-lua-bitop_1.0.2-5_amd64.deb ...
Unpacking lua-bitop:amd64 (1.0.2-5) ...
Selecting previously unselected package lua-cjson:amd64.
Preparing to unpack .../4-lua-cjson_2.1.0+dfsg-2.1_amd64.deb ...
Unpacking lua-cjson:amd64 (2.1.0+dfsg-2.1) ...
Selecting previously unselected package redis-tools.
Preparing to unpack .../5-redis-tools_5%3a6.0.16-1ubuntu1_amd64.deb ...
Unpacking redis-tools (5:6.0.16-1ubuntu1) ...
Setting up libjemalloc2:amd64 (5.2.1-4ubuntu1) ...
Setting up lua-cjson:amd64 (2.1.0+dfsg-2.1) ...
Setting up liblzf1:amd64 (3.6-3) ...
Setting up lua-bitop:amd64 (1.0.2-5) ...
Setting up liblua5.1-0:amd64 (5.1.5-8.1build4) ...
Setting up redis-tools (5:6.0.16-1ubuntu1) ...
Processing triggers for man-db (2.10.2-1) ...
Processing triggers for libc-bin (2.35-0ubuntu3.11) ...
(venv) user@user-HP-Pavilion-Gaming-Laptop:~/Web_Scraping_ChatBot/backend$ redis-cli ping
PONG
(venv) user@user-HP-Pavilion-Gaming-Laptop:~/Web_Scraping_ChatBot/backend$ Starting Redis-Commander at http://127.0.0.1:6379
Starting: command not found
(venv) user@user-HP-Pavilion-Gaming-Laptop:~/Web_Scraping_ChatBot/backend$ Starting Redis-Commander at http://127.0.0.1:6379
(venv) user@user-HP-Pavilion-Gaming-Laptop:~/Web_Scraping_ChatBot/backend$ npm install -g redis-commander
npm warn deprecated inflight@1.0.6: This module is not supported, and leaks memory. Do not use it. Check out lru-cache if you want a good and tested way to coalesce async requests by a key value, which is much more comprehensive and powerful.
npm warn deprecated glob@7.2.3: Glob versions prior to v9 are no longer supported
npm warn deprecated are-we-there-yet@2.0.0: This package is no longer supported.
npm warn deprecated gauge@3.0.2: This package is no longer supported.
npm warn deprecated npmlog@5.0.1: This package is no longer supported.
npm warn deprecated rimraf@3.0.2: Rimraf versions prior to v4 are no longer supported
npm warn deprecated lodash.isequal@4.5.0: This package is deprecated. Use require('node:util').isDeepStrictEqual instead.

added 192 packages in 9s

20 packages are looking for funding
  run `npm fund` for details
npm notice
npm notice New major version of npm available! 10.8.2 -> 11.6.2
npm notice Changelog: https://github.com/npm/cli/releases/tag/v11.6.2
npm notice To update run: npm install -g npm@11.6.2
npm notice
(venv) user@user-HP-Pavilion-Gaming-Laptop:~/Web_Scraping_ChatBot/backend$ echo $PATH
/home/user/Web_Scraping_ChatBot/backend/venv/bin:/home/user/miniconda3/condabin:/home/user/.nvm/versions/node/v18.20.8/bin:/home/user/.local/share/pnpm:/home/user/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin
(venv) user@user-HP-Pavilion-Gaming-Laptop:~/Web_Scraping_ChatBot/backend$ npx redis-commander
Starting with NODE_ENV=undefined and config NODE_APP_INSTANCE=undefined
Using scan instead of keys
No Save: false
listening on 0.0.0.0:8081
access with browser at http://127.0.0.1:8081
Redis Connection localhost:6379 using Redis DB #0
loading keys by prefix ""
found 0 keys for "" on node 0 (localhost:6379)

 -->
