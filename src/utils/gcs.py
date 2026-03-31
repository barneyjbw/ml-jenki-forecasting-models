import io
from typing import Iterator

from google.cloud import storage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.utils.logging import get_logger

logger = get_logger(__name__)

_client: storage.Client | None = None


def get_client() -> storage.Client:
    global _client
    if _client is None:
        _client = storage.Client()
    return _client


def _parse_uri(uri: str) -> tuple[str, str]:
    """Split gs://bucket/path into (bucket, path)."""
    assert uri.startswith("gs://"), f"Expected gs:// URI, got: {uri}"
    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    blob_path = parts[1] if len(parts) > 1 else ""
    return bucket, blob_path


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def download_bytes(uri: str) -> bytes:
    bucket_name, blob_path = _parse_uri(uri)
    bucket = get_client().bucket(bucket_name)
    blob = bucket.blob(blob_path)
    logger.info(f"Downloading {uri}")
    return blob.download_as_bytes()


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def upload_bytes(data: bytes, uri: str, content_type: str = "application/octet-stream") -> None:
    bucket_name, blob_path = _parse_uri(uri)
    bucket = get_client().bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_file(io.BytesIO(data), content_type=content_type)
    logger.info(f"Uploaded {len(data)} bytes to {uri}")


def list_blobs(uri: str) -> Iterator[str]:
    """Yield gs:// URIs for all blobs under a GCS prefix."""
    bucket_name, prefix = _parse_uri(uri)
    client = get_client()
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        yield f"gs://{bucket_name}/{blob.name}"
