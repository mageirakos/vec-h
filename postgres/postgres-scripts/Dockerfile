FROM docker.io/pgvector/pgvector:pg17

USER root

# Python + deps for load script and utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages \
    psycopg[binary] pgvector pyarrow numpy tqdm pandas

# Custom entrypoint for standalone postgres lifecycle (no docker-compose orchestration)
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY flush_caches.py /usr/local/bin/flush_caches.py
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 5432

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["postgres"]
