FROM python:3.11-slim

WORKDIR /app

# LightGBM requires libgomp (OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy source first so pip install can find the package
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

COPY schema.sql .
COPY data/ data/

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
