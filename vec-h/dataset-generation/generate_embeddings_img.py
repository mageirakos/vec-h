#!/usr/bin/env python3
"""
Generate image embeddings for Amazon product images using SigLIP2 (default) or Jina v4.

Reads all *.jsonl.gz files from --input-dir and writes partitioned parquet to --output-dir.

Usage:
    python generate_embeddings_img.py --input-dir data/amazon-23/raw/meta --output-dir data/amazon-23/parquet/siglip/images
    python generate_embeddings_img.py --input-dir data/amazon-23/raw/meta --output-dir data/amazon-23/parquet/jina/images --model-name jina-v4

Long-running:
    nohup python generate_embeddings_img.py ... >> embedding_img.log 2>&1 &
"""

import argparse
import gzip
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate image embeddings for Amazon product images.")
    parser.add_argument("--input-dir", required=True, help="Directory containing meta *.jsonl.gz files")
    parser.add_argument("--output-dir", required=True, help="Directory to write parquet files")
    parser.add_argument("--model-name", default="google/siglip2-so400m-patch14-384",
                        help="Model to use (default: google/siglip2-so400m-patch14-384). Use 'jina-v4' for Jina.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--no-fp16", action="store_true", help="Disable fp16 precision")
    parser.add_argument("--partition-size-gb", type=float, default=1.0,
                        help="Target size per partition file in GB (default: 1.0)")
    parser.add_argument("--rows-per-rowgroup", type=int, default=100000,
                        help="Rows per parquet row group (default: 100000)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Embedding generators
# ---------------------------------------------------------------------------

class EmbeddingGenerator:
    """Base class. Subclasses implement _load_model() and _encode()."""

    def __init__(self, batch_size: int = 4, use_fp16: bool = True):
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.model = None
        self.device: Optional[str] = None

    def _setup_device(self) -> bool:
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        logger.info("Using device: %s", self.device)
        return self.device == "cuda"

    def _load_model(self):
        raise NotImplementedError

    def _encode(self, inputs) -> torch.Tensor:
        raise NotImplementedError

    def encode_batch(self, inputs) -> torch.Tensor:
        if self.model is None:
            self._load_model()
        return self._encode(inputs)


class EmbeddingGeneratorQwen(EmbeddingGenerator):

    def __init__(self, batch_size: int = 8, use_fp16: bool = True,
                 model_size: str = "0.6B", max_seq_length: Optional[int] = None):
        super().__init__(batch_size, use_fp16)
        self.model_size = model_size
        self.max_seq_length = max_seq_length

    def _load_model(self):
        valid_sizes = ["0.6B", "4B", "8B"]
        if self.model_size not in valid_sizes:
            raise ValueError(f"Invalid model_size '{self.model_size}'. Must be one of {valid_sizes}")
        use_cuda = self._setup_device()
        model_id = f"Qwen/Qwen3-Embedding-{self.model_size}"
        logger.info("Loading %s...", model_id)
        if use_cuda and self.use_fp16:
            model_kwargs = {"attn_implementation": "sdpa", "device_map": "auto", "dtype": torch.float16}
        elif use_cuda:
            model_kwargs = {"device_map": "auto"}
        else:
            # MPS or CPU: let SentenceTransformer handle device placement automatically
            model_kwargs = {}
        self.model = SentenceTransformer(model_id, model_kwargs=model_kwargs,
                                         processor_kwargs={"padding_side": "left"})
        if self.max_seq_length is not None:
            self.model.max_seq_length = self.max_seq_length
        logger.info("Qwen %s loaded on %s", self.model_size, self.device)

    def _encode(self, inputs: List[str]) -> torch.Tensor:
        with torch.no_grad():
            return self.model.encode(
                inputs, batch_size=self.batch_size, convert_to_tensor=True,
                device=self.device, show_progress_bar=False,
            )


class EmbeddingGeneratorJinaV4(EmbeddingGenerator):

    def _load_model(self):
        use_cuda = self._setup_device()
        model_id = "jinaai/jina-embeddings-v4"
        logger.info("Loading %s...", model_id)
        if use_cuda and self.use_fp16:
            model_kwargs = {"device_map": "auto", "dtype": torch.float16}
        elif use_cuda:
            model_kwargs = {"device_map": "auto"}
        else:
            model_kwargs = {}  # Let SentenceTransformer auto-detect (MPS or CPU)
        self.model = SentenceTransformer(model_id, model_kwargs=model_kwargs, trust_remote_code=True)
        logger.info("Jina v4 loaded on %s", self.device)

    def _encode(self, inputs: List[str]) -> torch.Tensor:
        with torch.no_grad():
            return self.model.encode(
                inputs, task="retrieval", batch_size=self.batch_size,
                convert_to_tensor=True, device=self.device, show_progress_bar=False,
            )


class EmbeddingGeneratorSiglip(EmbeddingGenerator):

    def __init__(self, batch_size: int = 4, use_fp16: bool = True,
                 model_name: str = "google/siglip2-so400m-patch14-384"):
        super().__init__(batch_size, use_fp16)
        self.model_name = model_name
        self.processor = None

    def _load_model(self):
        use_cuda = self._setup_device()
        logger.info("Loading SigLIP model: %s", self.model_name)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        if use_cuda:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                device_map="auto",
                dtype=torch.float16 if self.use_fp16 else torch.float32,
            )
        else:
            # MPS/CPU: device_map="auto" doesn't support MPS; load then move
            self.model = AutoModel.from_pretrained(
                self.model_name,
                dtype=torch.float32,  # float16 has limited op support on MPS
            ).to(self.device)
        logger.info("SigLIP loaded on %s", self.device)

    def _encode(self, inputs: List[Image.Image]) -> torch.Tensor:
        """Encode a batch of pre-fetched PIL Images."""
        with torch.no_grad():
            model_inputs = self.processor(images=inputs, return_tensors="pt").to(self.device)
            output = self.model.get_image_features(**model_inputs)
            # transformers ≥5.x may return a ModelOutput instead of a tensor
            if isinstance(output, torch.Tensor):
                image_features = output
            else:
                # BaseModelOutputWithPooling: pooler_output is the [CLS] embedding
                image_features = output.pooler_output
            return image_features / image_features.norm(p=2, dim=-1, keepdim=True)


def make_generator(model_name: str, batch_size: int, use_fp16: bool,
                   model_size: str = "0.6B", max_seq_length: Optional[int] = None) -> EmbeddingGenerator:
    name_lower = model_name.lower()
    if "siglip" in name_lower:
        return EmbeddingGeneratorSiglip(batch_size=batch_size, use_fp16=use_fp16, model_name=model_name)
    elif name_lower == "jina-v4":
        return EmbeddingGeneratorJinaV4(batch_size=batch_size, use_fp16=use_fp16)
    elif name_lower == "qwen":
        return EmbeddingGeneratorQwen(batch_size=batch_size, use_fp16=use_fp16,
                                      model_size=model_size, max_seq_length=max_seq_length)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Use 'qwen', 'jina-v4', or a siglip model ID.")


# ---------------------------------------------------------------------------
# Image fetching
# ---------------------------------------------------------------------------

def _fetch_image(url_or_path: str, timeout: int = 10) -> Optional[Image.Image]:
    try:
        if url_or_path.startswith('http'):
            resp = requests.get(url_or_path, timeout=timeout)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert('RGB')
        else:
            return Image.open(url_or_path).convert('RGB')
    except Exception as e:
        logger.warning("Failed to load image %s: %s", url_or_path, e)
        return None


def _fetch_images_parallel(urls: List[str], max_workers: int = 16) -> List[Optional[Image.Image]]:
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_fetch_image, urls))


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def stream_images_from_meta_jsonl_gz(filepath: Path) -> Generator[Dict, None, None]:
    """Stream image records from a meta .jsonl.gz file, exploding each item's image list."""
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                parent_asin = item.get('parent_asin')
                images = item.get('images', [])
                if not parent_asin or not images:
                    continue
                for img_idx, image in enumerate(images):
                    image_url = image.get('large') or image.get('hi_res') or image.get('thumb')
                    if not image_url:
                        continue
                    yield {
                        'parent_asin': parent_asin,
                        'image_url': image_url,
                        'variant': image.get('variant', 'UNKNOWN'),
                        'image_index': img_idx,
                    }
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse line %d in %s: %s", line_idx, filepath.name, e)


# ---------------------------------------------------------------------------
# Parquet schema and shared infrastructure (imported by generate_embeddings_review.py)
# ---------------------------------------------------------------------------

def get_schema() -> pa.Schema:
    return pa.schema([
        pa.field('parent_asin', pa.string()),
        pa.field('image_url',   pa.string()),
        pa.field('variant',     pa.string()),
        pa.field('image_index', pa.int32()),
        pa.field('embedding',   pa.large_list(pa.float32())),
    ])


def create_pyarrow_table(metadata_list: List[Dict], embeddings_np: np.ndarray, embedding_dim: int) -> pa.Table:
    n_rows = len(metadata_list)
    flat_embeddings = embeddings_np.flatten().astype(np.float32)
    offsets = np.arange(0, (n_rows + 1) * embedding_dim, embedding_dim, dtype=np.int64)
    embedding_array = pa.LargeListArray.from_arrays(
        pa.array(offsets, type=pa.int64()),
        pa.array(flat_embeddings, type=pa.float32()),
    )
    return pa.table({
        'parent_asin': pa.array([m['parent_asin'] for m in metadata_list], type=pa.string()),
        'image_url':   pa.array([m['image_url']   for m in metadata_list], type=pa.string()),
        'variant':     pa.array([m['variant']      for m in metadata_list], type=pa.string()),
        'image_index': pa.array([m['image_index']  for m in metadata_list], type=pa.int32()),
        'embedding':   embedding_array,
    })


class PartitionedParquetWriter:
    """Writes embeddings to partitioned parquet files with buffering and automatic size-based rotation."""

    def __init__(self, base_path: Path, schema: pa.Schema,
                 target_size_gb: float = 1.0, rows_per_rowgroup: int = 100000,
                 table_factory: Optional[Callable] = None):
        self.base_path = Path(base_path)
        self.schema = schema
        self.target_size_bytes = int(target_size_gb * 1024 ** 3)
        self.rows_per_rowgroup = rows_per_rowgroup
        self.table_factory = table_factory or create_pyarrow_table
        self.current_partition = 0
        self.current_writer = None
        self.current_file_path = None
        self.buffer_metadata: List[Dict] = []
        self.buffer_embeddings: List[np.ndarray] = []
        self.buffer_size = 0

    def _open_new_partition(self):
        if self.current_writer is not None:
            self.current_writer.close()
        filename = f"part{self.current_partition:04d}.parquet"
        self.current_file_path = self.base_path / filename
        logger.info("Opening partition: %s", filename)
        self.current_writer = pq.ParquetWriter(
            self.current_file_path, self.schema,
            compression='snappy', use_dictionary=True, write_statistics=True, version='2.6',
        )
        self.current_partition += 1

    def _flush_buffer(self, embedding_dim: int):
        if self.buffer_size == 0:
            return
        table = self.table_factory(
            self.buffer_metadata, np.vstack(self.buffer_embeddings), embedding_dim
        )
        if self.current_writer is None:
            self._open_new_partition()
        self.current_writer.write_table(table)
        self.buffer_metadata, self.buffer_embeddings, self.buffer_size = [], [], 0
        if self.current_file_path.stat().st_size >= self.target_size_bytes:
            self._open_new_partition()

    def add_batch(self, metadata_list: List[Dict], embeddings_np: np.ndarray, embedding_dim: int):
        self.buffer_metadata.extend(metadata_list)
        self.buffer_embeddings.append(embeddings_np)
        self.buffer_size += len(metadata_list)
        if self.buffer_size >= self.rows_per_rowgroup:
            self._flush_buffer(embedding_dim)

    def close(self, embedding_dim: int):
        self._flush_buffer(embedding_dim)
        if self.current_writer is not None:
            self.current_writer.close()
            self.current_writer = None


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_images(
    input_dir: Path,
    output_dir: Path,
    model_name: str = "google/siglip2-so400m-patch14-384",
    batch_size: int = 4,
    use_fp16: bool = True,
    partition_size_gb: float = 1.0,
    rows_per_rowgroup: int = 100000,
):
    input_files = sorted(input_dir.glob("*.jsonl.gz"))
    if not input_files:
        logger.error("No *.jsonl.gz files found in %s", input_dir)
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Model: %s | Batch: %d | Files: %d", model_name, batch_size, len(input_files))
    for f in input_files:
        logger.info("  %s", f.name)

    generator = make_generator(model_name, batch_size=batch_size, use_fp16=use_fp16)
    writer = PartitionedParquetWriter(
        base_path=output_dir, schema=get_schema(),
        target_size_gb=partition_size_gb, rows_per_rowgroup=rows_per_rowgroup,
    )

    embedding_dim: Optional[int] = None
    n_images_failed = 0
    batch_inputs: List[str] = []
    batch_metadata: List[Dict] = []

    def _flush_batch():
        nonlocal embedding_dim, n_images_failed
        fetched = _fetch_images_parallel(batch_inputs)
        valid = [(img, meta) for img, meta in zip(fetched, batch_metadata) if img is not None]
        n_failed = len(batch_inputs) - len(valid)
        if n_failed:
            n_images_failed += n_failed
        if not valid:
            return
        encode_inputs, encode_metadata = map(list, zip(*valid))
        embeddings_np = generator.encode_batch(encode_inputs).cpu().float().numpy().astype(np.float32)
        if embedding_dim is None:
            embedding_dim = embeddings_np.shape[1]
            logger.info("Embedding dimension: %d", embedding_dim)
        writer.add_batch(encode_metadata, embeddings_np, embedding_dim)

    try:
        with tqdm(desc="images", unit="img") as pbar:
            for input_file in input_files:
                logger.info("Processing %s", input_file.name)
                for record in stream_images_from_meta_jsonl_gz(input_file):
                    batch_inputs.append(record['image_url'])
                    batch_metadata.append({
                        'parent_asin': record['parent_asin'],
                        'image_url':   record['image_url'],
                        'variant':     record['variant'],
                        'image_index': record['image_index'],
                    })

                    if len(batch_inputs) >= batch_size:
                        _flush_batch()
                        pbar.update(len(batch_inputs))
                        batch_inputs, batch_metadata = [], []

            if batch_inputs:
                _flush_batch()
                pbar.update(len(batch_inputs))

    except KeyboardInterrupt:
        logger.warning("Interrupted")
        if embedding_dim is not None:
            writer.close(embedding_dim)
        sys.exit(1)

    except Exception:
        logger.exception("Error during processing")
        if embedding_dim is not None:
            writer.close(embedding_dim)
        raise

    if embedding_dim is not None:
        writer.close(embedding_dim)

    if n_images_failed:
        logger.warning("Failed image downloads skipped: %d", n_images_failed)
    partition_files = sorted(output_dir.glob("part*.parquet"))
    total_size = sum(f.stat().st_size for f in partition_files)
    logger.info("Done. dim=%d | %d partitions | %.1f MB total",
                embedding_dim or 0, len(partition_files), total_size / 1024 / 1024)


def main():
    args = parse_args()
    process_images(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
        batch_size=args.batch_size,
        use_fp16=not args.no_fp16,
        partition_size_gb=args.partition_size_gb,
        rows_per_rowgroup=args.rows_per_rowgroup,
    )


if __name__ == "__main__":
    main()
