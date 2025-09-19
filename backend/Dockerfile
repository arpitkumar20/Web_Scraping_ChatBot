# # Use official lightweight Python image
# FROM python:3.11-slim

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# # Install system dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     gcc \
#     libpq-dev \
#     default-libmysqlclient-dev \
#     python3-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Set work directory
# WORKDIR /app

# # Copy requirements first for better caching
# COPY app/requirements.txt .

# # Upgrade pip and install dependencies
# RUN pip install --no-cache-dir --upgrade pip setuptools>=68.0.0 wheel>=0.41.0 \
#     && pip install --no-cache-dir -r requirements.txt

# # Copy project files
# COPY . .

# # Expose Flask default port
# EXPOSE 5000

# # Run the Flask app
# CMD ["python", "-m", "app.service"]

# Use official lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app  

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    default-libmysqlclient-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY app/requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools>=68.0.0 wheel>=0.41.0 \
    && pip install --no-cache-dir -r requirements.txt

COPY .env .

# Copy all project files (including .env)
COPY . .

# Expose Flask default port
EXPOSE 5000

# Run the Flask app as a module so imports work
CMD ["python", "-m", "app.service"]
