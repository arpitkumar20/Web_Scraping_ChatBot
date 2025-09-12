# FROM python:3.11-slim

# WORKDIR /app
# ENV PIP_NO_CACHE_DIR=1

# RUN apt-get update && apt-get install -y \
#     libglib2.0-0 libnss3 libatk1.0-0 libatk-bridge2.0-0 \
#     libcups2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
#     libxrandr2 libgbm1 libasound2 libpango-1.0-0 libgtk-3-0 \
#     && rm -rf /var/lib/apt/lists/*

# COPY backend/requirements.txt .
# RUN pip install -r requirements.txt

# # playwright browsers (chromium)
# RUN python -m playwright install --with-deps chromium

# COPY backend /app
# ENV PYTHONUNBUFFERED=1

# CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "app.wsgi:app"]