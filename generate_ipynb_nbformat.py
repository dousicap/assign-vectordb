#!/usr/bin/env python3
"""
Génère Academic_Research_Helper_Overlap.ipynb et le zip Academic_Research_Helper_Overlap.zip.
Le notebook est construit exclusivement via nbformat et contient l’overlap dans le chunking,
ainsi que l’architecture end-to-end load → clean → chunk with overlap → embed → store dans Qdrant → retrieve.
"""

from __future__ import annotations

import zipfile
import nbformat as nbf
from pathlib import Path

def build_notebook() -> nbf.NotebookNode:
    # Define a French description markdown cell
    md_intro = (
        "# Academic Research Helper avec overlap dans le chunking\n\n"
        "Objectif: ingérer des abstracts académiques synthétiques, les découper en morceaux avec overlap, "
        "les encoder (via EURI si une clé est fournie, fallback local sinon), les stocker dans Qdrant et "
        "permettre une recherche sémantique pour des requêtes comme « transformers in computer vision ». Le chunking "
        "inclut désormais un overlap entre chunks."
    )

    # Code cells: each block of logic from the previous notebook
    code1 = '''import os
import re
import requests
from typing import List, Dict, Any, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Configuration
VECTOR_SIZE = 128
CHUNK_MAX_CHARS = 600
CHUNK_OVERLAP_CHARS = 100  # overlap between consecutive chunks (in characters)
COLLECTION_NAME = "academic_papers"

# Qdrant host/port (override via environment if needed)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
'''

    code2 = '''def generate_dataset(n: int = 6) -> List[Dict[str, Any]]:
    """Génère un petit jeu de données synthétiques de papiers (id, title, abstract)."""
    base_papers = [
        {"id": "P1", "title": "Transformers in Computer Vision: A Survey",
         "abstract": "The transformer architecture has emerged as a powerful model for sequence modeling. This paper surveys transformer-based models in computer vision, including ViT, DeiT, and data-efficient variants. We discuss architectures, training regimes, and evaluation benchmarks."},
        {"id": "P2", "title": "Vision Transformers for Image Recognition",
         "abstract": "We examine Vision Transformers (ViT) architectures, patch embeddings, and how self-attention captures long-range dependencies in images. We compare with CNN-based baselines and discuss efficiency and scalability."},
        {"id": "P3", "title": "Self-Attention Mechanisms in Vision Tasks",
         "abstract": "Self-attention modules and their variants are applied to object detection, segmentation, and action recognition. We analyze computational trade-offs and show improvements on common benchmarks."},
        {"id": "P4", "title": "Transformers in Object Detection",
         "abstract": "Transformers extend detection pipelines with query-based decoding and cross-attention. This paper surveys DETR-like models and improvements such as Deformable DETR and query-based attention."},
        {"id": "P5", "title": "Efficient Transformers for Vision",
         "abstract": "We discuss efficiency techniques in transformers for vision, including sparse attention, kernel-based methods, and distillation strategies to reduce compute and memory footprints."},
        {"id": "P6", "title": "Multimodal Transformers for Vision-Language",
         "abstract": "Extending transformers to vision-language tasks, we review approaches like CLIP and ALIGN, focusing on alignment between text and image representations and zero-shot capabilities."},
    ]
    out = []
    for i in range(n):
        p = base_papers[i % len(base_papers)].copy()
        p["id"] = f"{p['id']}_{i}"
        out.append(p)
    return out
'''

    code3 = '''def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\\s+", " ", text)
    return text
'''

    code4 = '''def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap_chars: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    """Découpe le texte en chunks avec overlap entre chunks.

    Paramètres:
      max_chars: longueur maximale d'un chunk en caractères.
      overlap_chars: nombre de caractères qui se chevauchent entre chunks consécutifs.
    """
    if max_chars <= 0:
        return [text]
    chunks: List[str] = []
    if not text:
        return chunks
    i = 0
    n = len(text)
    while i < n:
        end = min(i + max_chars, n)
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        # Avancer en laissant un overlap de chars entre les chunks
        i = max(0, end - overlap_chars)
    return chunks
'''

    code5 = '''def generate_embeddings(text: str) -> List[float]:
    """
    Embedding function: utilise l’API EURI si clé présente; sinon, fallback local déterministe.
    """
    api_key = os.getenv("EURI_API_KEY")
    endpoint = os.getenv("EURI_EMBEDDING_ENDPOINT", "https://api.euri.ai/v1/embed")

    if api_key:
        try:
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {"text": text}
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            embedding = None
            if isinstance(data, dict):
                embedding = data.get("embedding") or data.get("vector") or data.get("embeddings")
            if isinstance(embedding, list):
                return embedding
        except Exception as e:
            print(f"[EURI] Embedding API failed: {e}")

    # Fallback deterministe (128-dim)
    dim = VECTOR_SIZE
    vec = [0.0] * dim
    for idx, ch in enumerate(text):
        vec[idx % dim] += (ord(ch) / 255.0)
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec
'''

    code6 = '''def ensure_collection(client: QdrantClient, name: str) -> None:
    try:
        client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        print(f"Collection '{name}' recreated.")
    except Exception:
        print(f"Collection '{name}' already exists or could not be recreated.")
'''

    code7 = '''def build_and_store_dataset(papers: List[Dict[str, Any]], client: QdrantClient, collection_name: str = COLLECTION_NAME) -> List[str]:
    """Ingest papers into Qdrant: chunk + embed + store. Retourne les IDs des points."""
    points: List[Dict[str, Any]] = []
    point_ids: List[str] = []
    for paper in papers:
        abstract = paper.get("abstract", "")
        clean = clean_text(abstract)
        chunks = chunk_text(clean, max_chars=CHUNK_MAX_CHARS, overlap_chars=CHUNK_OVERLAP_CHARS)
        for cid, chunk in enumerate(chunks):
            vec = generate_embeddings(chunk)
            point_id = f"{paper['id']}_{cid}"
            payload = {
                "paper_id": paper["id"],
                "title": paper.get("title"),
                "chunk_id": cid,
                "text": chunk
            }
            points.append({
                "id": point_id,
                "vector": vec,
                "payload": payload
            })
            point_ids.append(point_id)
    if not points:
        return []
    client.upsert(collection_name=collection_name, points=points)
    return point_ids
'''

    code8 = '''def _extract_payload_score(res: Any) -> Tuple[Dict[str, Any], float]:
    payload = getattr(res, "payload", None)
    score = getattr(res, "score", None)
    if payload is None and isinstance(res, dict):
        payload = res.get("payload", {})
        score = res.get("score", 0.0)
    return payload if payload is not None else {}, float(score if score is not None else 0.0)

def retrieve_papers(query: str, client: QdrantClient, collection_name: str = COLLECTION_NAME, top_k: int = 5) -> List[Dict[str, Any]]:
    vec = generate_embeddings(query)
    results = client.search(collection_name=collection_name, query_vector=vec, top=top_k, with_payload=True)
    hits: List[Dict[str, Any]] = []
    for r in results:
        payload, score = _extract_payload_score(r)
        title = payload.get("title")
        paper_id = payload.get("paper_id")
        chunk_id = payload.get("chunk_id")
        text = payload.get("text", "")
        hits.append({
            "score": score,
            "paper_id": paper_id,
            "title": title,
            "chunk_id": chunk_id,
            "text_snippet": text[:200] + ("..." if len(text) > 200 else "")
        })
    return hits
'''

    code9 = '''def run_demo():
    # Init Qdrant client
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Ensure collection exists/established
    ensure_collection(client, COLLECTION_NAME)

    # 1) Generate dataset
    papers = generate_dataset(n=6)

    # 2) Build and store (chunk + embed + store)
    print("Ingesting papers into Qdrant...")
    stored_ids = build_and_store_dataset(papers, client, COLLECTION_NAME)
    print(f"Stored {len(stored_ids)} chunks across {len(papers)} papers.")

    # 3) Semantic query example
    query = "transformers in computer vision"
    print(f"\nQuery: {query}")
    results = retrieve_papers(query, client, COLLECTION_NAME, top_k=5)
    print("Top results:")
    for r in results:
        print(f"- Paper ID: {r['paper_id']}, Title: {r['title']}, Score: {r['score']:.4f}")
        print(f"  Snippet: {r['text_snippet']}\n")

# Execute the demo when running this notebook cell
run_demo()
'''

    # Build the notebook
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(md_intro),
        nbf.v4.new_code_cell(code1),
        nbf.v4.new_code_cell(code2),
        nbf.v4.new_code_cell(code3),
        nbf.v4.new_code_cell(code4),
        nbf.v4.new_code_cell(code5),
        nbf.v4.new_code_cell(code6),
        nbf.v4.new_code_cell(code7),
        nbf.v4.new_code_cell(code8),
        nbf.v4.new_code_cell(code9),
    ]
    return nb

def main():
    nb = build_notebook()
    out_ipynb = Path("Academic_Research_Helper_Overlap.ipynb")
    # Write notebook using nbformat
    with open(out_ipynb, "w", encoding="utf-8") as f:
        nbf.write(nb, f, version=4)

    # Create zip archive containing the notebook
    zip_path = Path("Academic_Research_Helper_Overlap.zip")
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_ipynb, arcname=out_ipynb.name)

    print(f"Wrote {out_ipynb}")
    print(f"Created {zip_path}")

if __name__ == "__main__":
    main()