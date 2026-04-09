# 1. Pull the CUDA 12.8 Devel image
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# 2. Set the working directory
WORKDIR /app

# 3. Install system tools (Added 'curl' for the uv installer)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-dev \
    python3-venv \
    ffmpeg \
    git \
    unzip \
    nano \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Install 'uv' via the official standalone script
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 5. Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# 6. Install huggingface-cli cleanly using uv's tool manager
RUN uv tool install huggingface_hub[cli]

# 7. Copy project files
COPY . /app

# 8. Set the Git LFS environment variable globally
ENV GIT_LFS_SKIP_SMUDGE=1

# 9. Sync the project! 
# Because of the updated pyproject.toml, this natively pulls cu128 PyTorch
RUN uv sync --all-groups

# 10. Automatically use the virtual environment for all future commands/logins
ENV PATH="/app/.venv/bin:$PATH"

# 11. Compile flash_attn
# Because the venv is activated and synced, this builds perfectly against cu128 torch
RUN uv pip install flash-attn --no-build-isolation

# 12. Default command
CMD ["/bin/bash"]