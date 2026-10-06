# syntax=docker/dockerfile:1.7
# ---- builder: resolve and install dependencies into a venv --------------------------------
# Optional: build behind a TLS-intercepting proxy by passing your CA bundle directory, e.g.
#   CA_CERTS_DIR=/etc/ssl/certs docker compose build      (a file named ca-certificates.crt must exist there)
FROM scratch AS certs

FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY --from=certs / /certs/
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Dependencies first (cached until pyproject.toml changes), source afterwards.
COPY backend/pyproject.toml ./pyproject.toml
RUN python - > /tmp/requirements.txt <<'PY'
import tomllib
print("\n".join(tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]))
PY
RUN CA=/certs/ca-certificates.crt; if [ -f "$CA" ]; then export PIP_CERT=$CA SSL_CERT_FILE=$CA REQUESTS_CA_BUNDLE=$CA; fi; \
    pip install -r /tmp/requirements.txt
# Bake the (small) embedding models into the image so the container starts offline and
# reproducibly. (fastembed's offline BM25 loader reads <cache>/bm25/*.txt, hence the copy below.) Disable with --build-arg PRELOAD_MODELS=false to download on first start instead.
ARG PRELOAD_MODELS=true
ARG DENSE_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ARG SPARSE_MODEL=Qdrant/bm25
RUN CA=/certs/ca-certificates.crt; if [ -f "$CA" ]; then export SSL_CERT_FILE=$CA REQUESTS_CA_BUNDLE=$CA; fi; \
    if [ "$PRELOAD_MODELS" = "true" ]; then \
      python -c "from fastembed import TextEmbedding, SparseTextEmbedding; \
list(TextEmbedding('$DENSE_MODEL', cache_dir='/models').embed(['warm up'])); \
list(SparseTextEmbedding('$SPARSE_MODEL', cache_dir='/models').embed(['warm up']))" \
      && mkdir -p /models/bm25 && cp -rL /models/models--Qdrant--bm25/snapshots/*/. /models/bm25/ \
      && echo 'verifying the cache loads fully OFFLINE...' \
      && HF_HUB_OFFLINE=1 python -c "from fastembed import TextEmbedding, SparseTextEmbedding; \
list(TextEmbedding('$DENSE_MODEL', cache_dir='/models').embed(['x'])); \
list(SparseTextEmbedding('$SPARSE_MODEL', cache_dir='/models').embed(['x'])); print('offline model cache OK')"; \
    else mkdir -p /models; fi

COPY backend/src ./src
RUN CA=/certs/ca-certificates.crt; if [ -f "$CA" ]; then export PIP_CERT=$CA SSL_CERT_FILE=$CA REQUESTS_CA_BUNDLE=$CA; fi; \
    pip install --no-deps .


# ---- runtime ------------------------------------------------------------------------------
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PROJECT_ROOT=/app \
    EMBEDDING_CACHE_DIR=/models \
    LOG_FORMAT=json
# Models are baked into the image: never reach out to the HF hub at start-up.
# (Build with --build-arg HF_OFFLINE=0 together with PRELOAD_MODELS=false to download at runtime.)
ARG HF_OFFLINE=1
ENV HF_HUB_OFFLINE=${HF_OFFLINE}
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=app:app /models /models
COPY --chown=app:app data/knowledge_base ./data/knowledge_base
RUN mkdir -p /app/data/runtime && chown -R app:app /app/data /models
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=4).status == 200 else 1)"
# Optional runtime CA bundle (TLS-intercepting proxies): mount it at /certs/ca-certificates.crt.
CMD ["sh", "-c", "if [ -f /certs/ca-certificates.crt ]; then export SSL_CERT_FILE=/certs/ca-certificates.crt REQUESTS_CA_BUNDLE=/certs/ca-certificates.crt; fi; exec uvicorn company_brain.api.app:create_app --factory --host 0.0.0.0 --port 8000"]
