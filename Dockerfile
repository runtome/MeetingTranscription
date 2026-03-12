FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel

WORKDIR /workspace

# Install system packages
RUN apt-get update && \
    apt-get install -y ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir --break-system-packages \
    openai-whisper>=20231117 \
    transformers>=4.36.0 \
    numpy>=1.24.0

# Copy script (optional if you mount folder in Apptainer)
# COPY batch_to_csv.py .

# CMD ["python", "batch_to_csv.py"]