# Single-stage image: a pure-Python pipeline with C-extension wheels gains
# nothing from a separate build stage — the runtime image needs the same things.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Scrapy needs a working libxml2/libxslt; we install the runtime headers via
# the binary wheels (lxml ships them), so only minimal build tools are required.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the manifest first so the dependency layer is cached as long as
# pyproject.toml does not change.
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

# Bake the NLTK corpora into the image so the container does not need network
# access at runtime. The bootstrap script is idempotent and logs its progress.
RUN python -m wiki_recommender.nlp.bootstrap

# Run as a non-root user; matplotlib and Scrapy do not need root.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /app/plots /home/app/nltk_data \
    && cp -r /root/nltk_data/* /home/app/nltk_data/ 2>/dev/null || true \
    && chown -R app:app /app /home/app/nltk_data
USER app
ENV NLTK_DATA=/home/app/nltk_data

ENTRYPOINT ["python", "-m", "wiki_recommender"]
CMD ["--help"]
