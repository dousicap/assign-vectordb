# E-commerce Product Search

This project implements an end-to-end product search using **text**, **image**, and **speech**.

## Features
- Text search using embeddings
- Image search using CLIP-like embeddings (via Euri API)
- Speech search (speech-to-text → embedding search)
- Vector database powered by Qdrant
- UI with Gradio

## Setup
```bash
pip install gradio qdrant-client euri speechrecognition
docker run -p 6333:6333 qdrant/qdrant
export EURI_API_KEY=your_key_here
```
Run the notebook in VS Code or Jupyter.
