#!/bin/bash
# EC2 user-data bootstrap for code-modernizer (CPU-only, Ubuntu 22.04/24.04).
# Paste this into "User data" (Advanced details) when launching the instance.
# Installs Docker, Ollama, Python, and pulls the model — everything that
# doesn't depend on the project code, which you scp over separately.
set -euxo pipefail

# --- Docker ---
apt-get update -y
apt-get install -y ca-certificates curl gnupg python3.12-venv git
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin

usermod -aG docker ubuntu

# --- Ollama (CPU inference) ---
curl -fsSL https://ollama.com/install.sh | sh
systemctl enable ollama
systemctl start ollama
sleep 5
# Pull in the background — a 4.7GB download shouldn't block instance boot
sudo -u ubuntu ollama pull qwen2.5-coder:7b &

# --- App directory (code lands here via scp) ---
mkdir -p /home/ubuntu/code-modernizer
chown ubuntu:ubuntu /home/ubuntu/code-modernizer

echo "Bootstrap complete. Model pull continues in background — check with: ollama list" \
  > /home/ubuntu/BOOTSTRAP_DONE
chown ubuntu:ubuntu /home/ubuntu/BOOTSTRAP_DONE
