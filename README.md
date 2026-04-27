# Minimal RAG Solution — HPE Customer Engineering Interview

A self-contained Retrieval-Augmented Generation system that runs entirely
on CPU with no external services. Everything installs via pip and runs
inside a single Jupyter notebook.

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/iminierai-aig/rag-demo.git
cd rag-demo

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook rag_solution.ipynb
```

Run cells sequentially. The notebook will:

1. Download models automatically from Hugging Face (~3.5 GB first run)
2. Create a sample dataset if `./data/` is empty
3. Index documents into a local ChromaDB vector store
4. Demonstrate the full RAG pipeline with interactive queries

## Requirements

- **Python** 3.10+
- **RAM** 8 GB minimum (16 GB recommended)
- **Disk** ~5 GB for models and vector store
- **GPU** Not required — runs on CPU

## Architecture

See `architecture.md` for component details and design rationale.

## Technology Stack

|Component|Technology|Why|
|---|---|---|
|Knowledge Base|ChromaDB (persistent)|Zero-config, embedded, Python-native|
|Embeddings|`all-MiniLM-L6-v2` (sentence-transformers)|Fast CPU inference, 384-dim, proven quality|
|Generation|`HuggingFaceTB/SmolLM2-1.7B-Instruct`|1.7B params, Apache 2.0, runs on CPU, strong instruction-following|
|Document Loading|LangChain (PyPDF, TextLoader)|Clean abstractions for PDF/TXT ingestion|
|Chunking|LangChain RecursiveCharacterTextSplitter|Handles semantic boundaries well|

## Deliverables

- `rag_solution.ipynb` — Interactive notebook demonstrating the full RAG pipeline
- `slides/` — Presentation deck with architecture diagram and design rationale
- `architecture.md` — Detailed component documentation