# CISCaRL Live — backend for Hugging Face Spaces (Docker)
FROM python:3.11-slim

# HF Spaces runs as a non-root user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

COPY --chown=user requirements-web.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-web.txt

COPY --chown=user . .

EXPOSE 7860
# Honor $PORT (Render/Fly/Koyeb set it); default 7860 for HF Spaces.
CMD ["sh", "-c", "uvicorn webapp.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
