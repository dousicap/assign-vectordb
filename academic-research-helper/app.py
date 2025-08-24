import os
import re
import uuid
import math
import argparse
from pathlib import Path
from typing import List, Dict, Iterable, Tuple


from dotenv import load_dotenv
from rich import print as rprint
from rich.table import Table


# PDF parsing
from pypdf import PdfReader


# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
Distance,
VectorParams,
PointStruct,
)


# EURI embeddings
from euriai.embedding import EuriaiEmbeddingClient


# ---------- Utilities ----------


def clean_text(text: str) -> str:
# Basic cleanup: collapse whitespace, remove odd chars
text = text.replace("\x00", " ")
text = re.sub(r"\s+", " ", text)
return text.strip()


# Simple token-ish splitter (by words) to avoid extra deps
# Uses word count as a proxy for tokens


def chunk_text(text: str, chunk_tokens: int, overlap: int) -> List[str]:
words = text.split()
if not words:
return []


step = max(1, chunk_tokens - overlap)
chunks = []
for i in range(0, len(words), step):
chunk = words[i : i + chunk_tokens]
if not chunk:
break
chunks.append(" ".join(chunk))
return chunks


# ---------- EURI: Embedding function ----------


def generate_embeddings(texts: List[str], api_key: str, model: str | None = None) -> List[List[float]]:
"""
Batch embed with EURI API.
- texts: list of strings
- returns: list of embedding vectors (List[float])
"""
client = EuriaiEmbeddingClient(api_key=api_key) if not model else EuriaiEmbeddingClient(api_key=api_key, model=model)
# euriai client exposes .embed(text) one-by-one; batch for convenience
vectors = []
for t in texts:
v = client.embed(t)
p_ingest = sub.add