FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /workspace

COPY pyproject.toml .
COPY training.py .
COPY config/ ./config/
COPY src/ ./src/
RUN pip install -e .[visualization,logging]


