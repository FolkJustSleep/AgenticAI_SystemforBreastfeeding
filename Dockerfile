FROM python:3.11-slim

WORKDIR /app

# Install system deps needed for hnswlib, opencv, and pillow
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libjpeg62-turbo \
    libpng16-16 \
    zlib1g \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/

RUN pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]