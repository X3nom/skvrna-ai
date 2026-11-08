FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /work

# Instruct uv to use a dedicated venv inside the container, ignoring any host .venv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PYTHONUNBUFFERED=1
EXPOSE 8888

# Copy lockfiles first to utilize Docker build layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into the container's isolated /opt/venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Start Jupyter using the container's environment
CMD ["uv", "run", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
