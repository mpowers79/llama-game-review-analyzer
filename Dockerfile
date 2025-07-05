# Dockerfile
FROM python:3.10-slim-buster 

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        meson \
        ninja-build \
        libatlas-base-dev \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x /app/*.sh

# Expose the port that Streamlit runs on (default is 8501)
EXPOSE 8501

ENTRYPOINT ["sh"]