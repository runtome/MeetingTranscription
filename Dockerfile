FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Install ffmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (only what batch_to_csv.py needs)
RUN pip install --no-cache-dir \
    openai-whisper>=20231117 \
    transformers>=4.36.0 \
    numpy>=1.24.0

# Copy project files
COPY batch_to_csv.py .

CMD ["python", "batch_to_csv.py"]
