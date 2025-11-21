FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /eruptions

COPY . .

RUN git init && \
    git config user.email "docker@example.com" && \
    git config user.name "Docker User" && \
    git add . && \
    git commit -m "Initial commit inside Docker"

RUN pip install --no-cache-dir -r requirements.txt

RUN dvc init -f

CMD ["dvc", "repro"]