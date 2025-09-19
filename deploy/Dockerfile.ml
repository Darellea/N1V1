# Multi-stage build for N1V1 ML component
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt && \
    # Install additional ML dependencies
    pip install scikit-learn xgboost lightgbm tensorflow pandas numpy

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash n1v1

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY ml/ ./ml/
COPY utils/ ./utils/
COPY config.json ./

# Create directories for models and data
RUN mkdir -p models experiments logs

# Change ownership to non-root user
RUN chown -R n1v1:n1v1 /app

# Switch to non-root user
USER n1v1

# Health check - simple file existence check for ML service
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD test -f /app/models/.health || exit 1

# Expose port for potential ML serving API
EXPOSE 8080

# Default command - run training
CMD ["python", "-m", "ml.train"]
