FROM python:3.11-slim

WORKDIR /app

COPY asset-embedding-job/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY tools/ /app/tools/
COPY asset-embedding-job/index_asset_embeddings.py /app/asset-embedding-job/index_asset_embeddings.py

ENTRYPOINT ["python", "/app/asset-embedding-job/index_asset_embeddings.py"]

