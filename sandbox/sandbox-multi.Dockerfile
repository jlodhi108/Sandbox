FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-pip \
    php-cli \
    default-jdk-headless \
    curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g typescript ts-node \
    && pip install --no-cache-dir --break-system-packages semgrep hypothesis \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Local, offline rule file — NOT a semgrep registry config (`p/...`).
# Registry configs try to revalidate over the network on every run even
# when previously cached, and hang indefinitely (no fast refusal) inside
# a network_disabled container — confirmed by testing directly. A local
# YAML file has zero network dependency at scan time.
COPY security-rules.yaml /opt/security-rules.yaml

RUN useradd -m -u 1000 sandboxuser
RUN chmod 644 /opt/security-rules.yaml
WORKDIR /workspace
USER sandboxuser
