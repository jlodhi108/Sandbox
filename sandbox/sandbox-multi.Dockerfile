FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    php-cli \
    default-jdk-headless \
    curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g typescript ts-node \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 sandboxuser
WORKDIR /workspace
USER sandboxuser
