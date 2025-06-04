FROM python:3.9-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and activate Python virtual environment
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy requirements first for better layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy application files
COPY . ./

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Expose the port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]