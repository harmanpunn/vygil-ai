# Multi-stage Dockerfile for Vygil AI Application
FROM node:22-alpine AS base

# Install system dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    gcc \
    g++ \
    make \
    curl \
    bash \
    ca-certificates 

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Set default port for Railway
ENV PORT=5173

# Copy package files for dependency installation
COPY pyproject.toml uv.lock ./
COPY frontend/package*.json ./frontend/
COPY mcp-server/package*.json ./mcp-server/

# Install Node.js dependencies
RUN cd frontend && npm install
RUN cd mcp-server && npm install

# Copy source code
COPY . .

# Build MCP Server (just build, don't run)
RUN cd mcp-server && npm run build

# Set up Python environment
RUN uv sync
ENV PATH="/app/.venv/bin:$PATH"

# Create necessary directories
RUN mkdir -p /app/data/screenshots /app/logs

# Expose ports (only for services we're actually running)
EXPOSE 8000 5173

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start script
COPY start-services.sh /app/start-services.sh
RUN chmod +x /app/start-services.sh

CMD ["/app/start-services.sh"]