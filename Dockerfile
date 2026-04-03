FROM python:3.11-slim

WORKDIR /app

# Copy source first so pip install can find the package
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

COPY schema.sql .
COPY data/ data/

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
