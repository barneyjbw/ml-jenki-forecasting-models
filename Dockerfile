FROM python:3.11-slim

# Stan compilation requires build tools + OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-compile Stan models at build time so the container starts fast
RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan(cores=2, progress=False)"

COPY . .

ENV DATA_SOURCE=gcs
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "scripts.retrain"]
