# ==============================================================================
# Retail Command Center: Production Enterprise Dockerfile
# Multi-Stage Build: Isolated builder -> Minimal hardened non-root runtime
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build & Dependency Isolation (builder)
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

# Prevent interactive debconf prompts and Python bytecode bloat in build layer
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install minimal compilation toolchain and build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Establish isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies strictly within /opt/venv using --no-cache-dir
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt


# ------------------------------------------------------------------------------
# Stage 2: Hardened Production Runtime (runtime)
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# GitHub Container Registry (GHCR) OCI Build Arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION="1.0.0"
ARG REPO_URL="https://github.com/mmxco/retail-command-center"

# OCI standard container annotations for GitHub Packages / GHCR
LABEL org.opencontainers.image.title="Retail AI Pre-Sales Command Center" \
    org.opencontainers.image.description="Unified Enterprise Cockpit Integrating Account Discovery, Multi-Agent Replenishment Negotiation, and Balance Sheet Valuation Forensics." \
    org.opencontainers.image.url="${REPO_URL}" \
    org.opencontainers.image.source="${REPO_URL}" \
    org.opencontainers.image.documentation="${REPO_URL}/blob/main/README.md" \
    org.opencontainers.image.vendor="Retail AI Enterprise" \
    org.opencontainers.image.licenses="Apache-2.0" \
    org.opencontainers.image.version="${VERSION}" \
    org.opencontainers.image.revision="${VCS_REF}" \
    org.opencontainers.image.created="${BUILD_DATE}"

# Production container execution environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    PATH="/opt/venv/bin:$PATH"

# Install minimal OS-level system libraries for Playwright headless Chromium & healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libdrm2 \
    libgbm1 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxkbcommon0 \
    libxrandr2 \
    && rm -rf /var/lib/apt/lists/*

# Transfer pre-built virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Pre-populate dedicated Playwright browser cache directory with standalone Chromium
RUN mkdir -p /ms-playwright && \
    playwright install chromium

# Enforce enterprise security: Create unprivileged system group & user
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1001 -m -d /home/appuser appuser

# Set up production application workspace
WORKDIR /app

# Copy application artifacts with unprivileged user ownership
COPY --chown=appuser:appgroup . /app

# Re-enforce permissions across application tree, venv, and Playwright binaries
RUN chown -R appuser:appgroup /app /opt/venv /ms-playwright

# Drop root privileges and enforce non-root context
USER 1001:1001

# Expose default Streamlit port
EXPOSE 8501

# Container healthcheck verifying Streamlit server readiness via curl
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch unified Streamlit Command Center via exec entrypoint
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
