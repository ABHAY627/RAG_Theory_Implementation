# Retrieval Augmented Generation (RAG)

A two-stage RAG pipeline using LangChain, ChromaDB, and HuggingFace embeddings.

## How it works

```
docs/*.txt  →  ingestion_pipeline.py  →  db/chroma_db/   (vector store)
                                                ↓
                                     retrieval_pipeline.py  →  answers
```

1. **Ingestion** — loads `.txt` files from `docs/`, splits them into overlapping chunks, embeds them with `all-MiniLM-L6-v2`, and persists to ChromaDB.
2. **Retrieval** — loads the vector store and runs similarity search against a set of queries, printing the top-k matching chunks.

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> The embedding model (`all-MiniLM-L6-v2`) is downloaded automatically from HuggingFace on first run — no API key required.

### 3. (Optional) Configure `.env`

The `.env` file is pre-configured. If you plan to extend the pipeline with an LLM (e.g. OpenAI for generation), add your key:

```
OPENAI_API_KEY=sk-...
```

## Running the pipeline

### Step 1 — Ingest documents

```bash
python ingestion_pipeline.py
```

This reads every `.txt` file in `docs/`, chunks and embeds them, then saves the vector store to `db/chroma_db/`. Re-running skips ingestion if the store already exists.

### Step 2 — Retrieve

```bash
python retrieval_pipeline.py
```

Runs three example queries against the vector store and prints the top-3 matching chunks for each.

## Project structure

```
Retrieval_Augmented_Generation/
├── docs/
│   ├── google.txt
│   ├── microsoft.txt
│   └── nvidia.txt
├── db/
│   └── chroma_db/          # auto-created by ingestion
├── ingestion_pipeline.py
├── retrieval_pipeline.py
├── requirements.txt
└── .env
```

## Adding your own documents

Drop any `.txt` file into `docs/` and delete the `db/chroma_db/` folder, then re-run `ingestion_pipeline.py`.
