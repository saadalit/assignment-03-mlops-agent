FROM python:3.12.12-slim
WORKDIR /app
COPY requirements-inference.txt ./
RUN pip install --no-cache-dir --require-hashes -r requirements-inference.txt \
    && useradd --uid 10001 --create-home app
COPY --chown=app:app api ./api
COPY --chown=app:app models/current ./models/current
USER app
ENV MODEL_DIR=/app/models/current PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
