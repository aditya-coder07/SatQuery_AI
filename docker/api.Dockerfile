FROM python:3.11-slim

WORKDIR /app

# Copy requirements if present, otherwise just continue
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

COPY . .

CMD ["python", "-m", "uvicorn", "satquery.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
