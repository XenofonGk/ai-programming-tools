# Stage 1: train the model with the exact pinned dependencies the API will load it with.
FROM python:3.12-slim AS train
WORKDIR /src
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt pandas==3.0.6
COPY preprocess.py train.py ./
COPY data ./data
RUN python data/generate_dataset.py && python train.py

# Stage 2: runtime. No pandas, no compiler, no training data; non-root.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt
COPY app.py preprocess.py ./
COPY static ./static
COPY --from=train /src/model ./model
RUN useradd --system --uid 10001 --no-create-home app
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if b'healthy' in urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read() else 1)"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header", "--proxy-headers"]
