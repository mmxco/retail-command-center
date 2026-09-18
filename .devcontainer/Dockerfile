FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# 1. Install OS-level system dependencies for headless Playwright Chromium
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        curl \
        libnss3 \
        libnspr4 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libdrm2 \
        libxkbcommon0 \
        libxcomposite1 \
        libxdamage1 \
        libxrandr2 \
        libgbm1 \
        libpango-1.0-0 \
        libasound2 \
        fonts-liberation \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Create and permission Playwright browser directory for the vscode dev user
RUN mkdir -p /ms-playwright && chown -R vscode:vscode /ms-playwright

# 3. Pre-install Python dependencies during image build
COPY requirements.txt /tmp/pip-tmp/requirements.txt
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp

# 4. Install Playwright browser binary (Chromium) as vscode user
USER vscode
RUN playwright install chromium
USER root