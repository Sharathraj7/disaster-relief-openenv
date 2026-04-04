FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Environment defaults
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Start FastAPI server — use root app.py for HF Spaces compatibility,
# server.app:app is the OpenEnv entry_point used by validators.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
