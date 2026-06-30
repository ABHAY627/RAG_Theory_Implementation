<div align="center">

<!-- Animated Header Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Retrieval%20Augmented%20Generation&fontSize=40&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Build%20Smarter%20AI%20•%20Ground%20Your%20LLM%20in%20Facts&descAlignY=55&descSize=18" width="100%"/>

<br/>

<!-- Animated Typing SVG -->
<a href="https://github.com/yourusername/Retrieval_Augmented_Generation">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=00D9FF&center=true&vCenter=true&multiline=true&width=700&height=80&lines=📚+Load+Docs+→+🧩+Chunk+→+🔢+Embed+→+💾+Store+→+🔍+Retrieve;Powered+by+LangChain+%7C+ChromaDB+%7C+HuggingFace" alt="Typing SVG" />
</a>

<br/><br/>

<!-- Badges -->
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/LangChain-0.3.25-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/>
<img src="https://img.shields.io/badge/ChromaDB-0.6.3-FF6B35?style=for-the-badge&logo=databricks&logoColor=white"/>
<img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
<img src="https://img.shields.io/badge/Google-Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>

<br/><br/>

</div>

---

## 📖 Table of Contents

<details open>
<summary><b>Click to expand / collapse</b></summary>

- [🧠 What is RAG?](#-what-is-rag)
- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🛠️ Tech Stack](#️-tech-stack)
- [📦 Embedding Model](#-embedding-model)
- [📁 Project Structure](#-project-structure)
- [⚡ Quick Start](#-quick-start)
- [🔄 Pipeline Walkthrough](#-pipeline-walkthrough)
- [📄 Sample Documents](#-sample-documents)
- [⚙️ Configuration](#️-configuration)
- [🗺️ Roadmap](#️-roadmap)
- [🤝 Contributing](#-contributing)

</details>

---

## 🧠 What is RAG?

<div align="center">
<img src="https://img.shields.io/badge/Concept-Retrieval%20Augmented%20Generation-blueviolet?style=flat-square&logo=openai"/>
</div>

**Retrieval Augmented Generation (RAG)** is a technique that supercharges Large Language Models by giving them access to your own knowledge base — preventing hallucinations and keeping answers grounded in real data.

```
Without RAG:  User Question  →  LLM (limited knowledge)  →  Possibly wrong answer ❌
With RAG:     User Question  →  Vector Search  →  Relevant Context  →  LLM  →  Accurate answer ✅
```

Instead of retraining an expensive LLM on your data, RAG **retrieves** the most relevant document chunks at query time and injects them as context. The LLM then generates answers based on that real, up-to-date information.

---

## ✨ Features

<table>
<tr>
<td>

- 📂 **Auto Document Loading** — scans an entire directory of `.txt` files
- 🧩 **Intelligent Chunking** — overlapping windows preserve cross-chunk context
- 🔢 **Local Embeddings** — runs 100% offline, no API key needed for ingestion
- 💾 **Persistent Vector Store** — ChromaDB saves embeddings to disk; skip re-ingestion on restart

</td>
<td>

- 🔍 **Cosine Similarity Search** — HNSW index for fast approximate nearest-neighbour retrieval
- ♻️ **Smart Cache Check** — automatically detects existing vector store and skips re-processing
- 🔌 **LLM Ready** — plug in Google Gemini (or any LangChain LLM) for full Q&A generation
- 🪟 **Windows Friendly** — UTF-8 loader config handles Windows encoding edge cases

</td>
</tr>
</table>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                           │
│                                                                     │
│  docs/*.txt                                                         │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────┐    ┌──────────────────────┐                   │
│  │  DirectoryLoader │───▶│  CharacterTextSplitter│                  │
│  │  (TextLoader)    │    │  chunk_size=1000      │                  │
│  └─────────────────┘    │  chunk_overlap=200    │                  │
│                          └──────────┬───────────┘                  │
│                                     │                               │
│                                     ▼                               │
│                          ┌──────────────────────┐                  │
│                          │  HuggingFaceEmbeddings│                  │
│                          │  all-MiniLM-L6-v2     │                  │
│                          │  (384-dim vectors)    │                  │
│                          └──────────┬───────────┘                  │
│                                     │                               │
│                                     ▼                               │
│                          ┌──────────────────────┐                  │
│                          │      ChromaDB         │                  │
│                          │  HNSW cosine index   │                  │
│                          │  db/chroma_db/        │                  │
│                          └──────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL PIPELINE                           │
│                                                                     │
│  User Query                                                         │
│      │                                                              │
│      ▼                                                              │
│  ┌──────────────────────┐    ┌──────────────────────┐              │
│  │  HuggingFaceEmbeddings│───▶│   ChromaDB            │             │
│  │  (same model)         │    │   similarity_search   │             │
│  └──────────────────────┘    │   top-k chunks        │             │
│                               └──────────┬───────────┘             │
│                                          │                          │
│                                          ▼                          │
│                               ┌──────────────────────┐             │
│                               │  Retrieved Context    │             │
│                               │  (ready for LLM)      │             │
│                               └──────────────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Tool | Version | Purpose |
|-------|------|---------|---------|
| 🔗 **Orchestration** | [LangChain](https://python.langchain.com/) | `0.3.25` | Pipeline glue — loaders, splitters, chains |
| 🔗 **Community Tools** | langchain-community | `0.3.24` | `DirectoryLoader`, `TextLoader` |
| ✂️ **Text Splitting** | langchain-text-splitters | `0.3.8` | `CharacterTextSplitter` |
| 🤗 **Embeddings** | langchain-huggingface | `1.2.1` | Bridge to HuggingFace models |
| 💾 **Vector Store** | [ChromaDB](https://www.trychroma.com/) | `0.6.3` | Persistent local vector database |
| 🔗 **Chroma Bridge** | langchain-chroma | `1.1.0` | LangChain ↔ ChromaDB integration |
| 🤖 **Sentence Model** | [sentence-transformers](https://www.sbert.net/) | `3.4.1` | Loads and runs embedding models |
| 🌐 **LLM (optional)** | [Google Gemini](https://ai.google.dev/) | via `langchain-google-genai 2.1.4` | Generation layer (Q&A) |
| ⚙️ **Env Config** | python-dotenv | `1.1.0` | Loads API keys from `.env` |
| 🐍 **Runtime** | Python | `3.10+` | Core language |

</div>

---

## 📦 Embedding Model

<div align="center">
<img src="https://img.shields.io/badge/Model-all--MiniLM--L6--v2-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
<img src="https://img.shields.io/badge/Dimensions-384-blue?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Size-~80MB-green?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Runs-Offline-brightgreen?style=for-the-badge"/>
</div>

<br/>

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

```
Input Text  →  Tokenizer  →  MiniLM-L6 (6-layer transformer)  →  384-dim vector
```

| Property | Value |
|----------|-------|
| Architecture | Sentence-BERT (SBERT) |
| Layers | 6 transformer layers |
| Output Dimensions | 384 |
| Max Input Tokens | 256 tokens |
| Model Size | ~80 MB |
| License | Apache 2.0 |
| Download | Automatic on first run (HuggingFace Hub) |
| Requires API Key | ❌ No — fully local |

**Why this model?**
- Lightweight and fast — perfect for learning and local dev
- Strong semantic similarity performance (trained on 1B+ sentence pairs)
- No GPU required — runs well on CPU

**Vector Similarity Metric used:** Cosine Similarity (configured via `hnsw:space: cosine` in ChromaDB)

---

## 📁 Project Structure

```
Retrieval_Augmented_Generation/
│
├── 📂 docs/                        # Source documents for ingestion
│   ├── 📄 google.txt               # Google company overview
│   ├── 📄 microsoft.txt            # Microsoft company overview
│   └── 📄 nvidia.txt               # Nvidia company overview
│
├── 💾 db/
│   └── chroma_db/                  # Auto-created persistent vector store
│       ├── chroma.sqlite3          # ChromaDB metadata & collections
│       └── <uuid>/                 # HNSW index binary files
│           ├── data_level0.bin
│           ├── header.bin
│           ├── length.bin
│           └── link_lists.bin
│
├── 🐍 ingestion_pipeline.py        # Stage 1: Load → Chunk → Embed → Store
├── 🔍 retrieval_pipeline.py        # Stage 2: Query → Search → Return chunks
├── 📋 requirements.txt             # Pinned Python dependencies
├── 🔑 .env                         # API keys (gitignored)
├── 🚫 .gitignore                   # Protects secrets & large files
└── 📖 README.md                    # You are here!
```

---

## ⚡ Quick Start

### Prerequisites

- Python `3.10+`
- `pip`
- ~500 MB free disk space (for embedding model download)

---

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/Retrieval_Augmented_Generation.git
cd Retrieval_Augmented_Generation
```

### 2️⃣ Create & activate virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

> 💡 On first run, `sentence-transformers` will auto-download `all-MiniLM-L6-v2` (~80MB) from HuggingFace Hub. No account or API key needed.

### 4️⃣ Configure environment (optional)

```bash
# .env is already set up — add your Gemini key to enable LLM generation
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 5️⃣ Run ingestion pipeline

```bash
python ingestion_pipeline.py
```

**Expected output:**
```
=== RAG Document Ingestion Pipeline ===

Loading documents from 'docs'...
Loaded 3 document(s).

Splitting documents into chunks...
Created 9 chunk(s).

Creating embeddings and storing in ChromaDB...
Building vector store — this may take a moment...
Vector store saved to 'db/chroma_db' (9 vectors).

✅ Ingestion complete! Documents are ready for RAG queries.
```

### 6️⃣ Run retrieval pipeline

```bash
python retrieval_pipeline.py
```

---

## 🔄 Pipeline Walkthrough

<details>
<summary><b>📂 Stage 1 — Document Loading</b></summary>

```python
loader = DirectoryLoader(
    path="docs",
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},  # Windows-safe
)
documents = loader.load()
```

- Scans the `docs/` folder for all `.txt` files
- Each file becomes a `Document` object with `page_content` + `metadata` (source path)
- UTF-8 encoding specified explicitly to handle Windows codec issues

</details>

<details>
<summary><b>✂️ Stage 2 — Text Chunking</b></summary>

```python
text_splitter = CharacterTextSplitter(
    chunk_size=1000,     # max characters per chunk
    chunk_overlap=200,   # overlap between consecutive chunks
    separator="\n",      # split on newlines first
)
chunks = text_splitter.split_documents(documents)
```

**Why overlap?**
If a key sentence spans a chunk boundary, the `chunk_overlap=200` ensures it appears in both chunks — so retrieval never misses critical context.

```
Document:  [-------- chunk 1 --------][--- overlap ---][-------- chunk 2 --------]
```

</details>

<details>
<summary><b>🔢 Stage 3 — Embedding Generation</b></summary>

```python
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

Each chunk's text is passed through the `all-MiniLM-L6-v2` transformer model.
Output: a **384-dimensional float vector** that encodes the semantic meaning of the text.

Similar content → similar vectors → close in vector space.

</details>

<details>
<summary><b>💾 Stage 4 — Vector Store Persistence</b></summary>

```python
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="db/chroma_db",
    collection_metadata={"hnsw:space": "cosine"},
)
```

- ChromaDB stores vectors in an **HNSW (Hierarchical Navigable Small World)** index
- HNSW enables lightning-fast approximate nearest-neighbour search
- Data is persisted to disk — survives restarts without re-embedding
- **Cosine similarity** metric used (angle between vectors, ideal for text)

</details>

<details>
<summary><b>🔍 Stage 5 — Retrieval (Similarity Search)</b></summary>

```python
# Query is embedded with the SAME model used during ingestion
results = vectorstore.similarity_search(query, k=3)
```

At query time:
1. Your query string is embedded → 384-dim vector
2. ChromaDB searches for the `k` most similar vectors in the HNSW index
3. Returns the corresponding document chunks as context
4. These chunks can be passed to an LLM (e.g. Gemini) for answer generation

</details>

---

## 📄 Sample Documents

The `docs/` folder ships with three tech company overviews — perfect for testing retrieval:

| File | Topic | Key Facts |
|------|-------|-----------|
| `google.txt` | Google LLC | Founded 1998 by Larry Page & Sergey Brin, Stanford PhD students |
| `microsoft.txt` | Microsoft Corp | Founded 1975 by Bill Gates & Paul Allen, acquired GitHub 2018 |
| `nvidia.txt` | Nvidia Corp | Founded 1993 by Jensen Huang, GPU leader, CUDA platform |

**Sample queries you can test:**
```
"Who founded Google?"
"When did Microsoft acquire GitHub?"
"What is CUDA?"
"Which company focuses on GPU computing?"
```

---

## ⚙️ Configuration

### Chunking Parameters

Edit in `ingestion_pipeline.py` → `split_documents()`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `chunk_size` | `1000` | Larger = more context per chunk, fewer chunks |
| `chunk_overlap` | `200` | Larger = less chance of missing boundary info |
| `separator` | `"\n"` | Character used to prefer split points |

### Retrieval Parameters

Edit in `retrieval_pipeline.py` → `similarity_search()`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `k` | `3` | Number of chunks returned per query |

### Switching Embedding Models

```python
# Faster, smaller (for prototyping)
HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")      # 384-dim, ~80MB

# Higher quality (for production)
HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")      # 768-dim, ~420MB

# Multilingual
HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
```

> ⚠️ If you change the embedding model, delete `db/chroma_db/` and re-run ingestion — dimensions must match!

### Adding Your Own Documents

1. Drop any `.txt` file into `docs/`
2. Delete the existing vector store: `rmdir /s /q db\chroma_db` (Windows)
3. Re-run: `python ingestion_pipeline.py`

---

## 🗺️ Roadmap

- [x] Document ingestion pipeline (load → chunk → embed → store)
- [x] Retrieval pipeline (query → similarity search → top-k chunks)
- [x] Persistent ChromaDB vector store
- [x] Smart cache detection (skip re-ingestion)
- [ ] Full Q&A chain with Google Gemini
- [ ] Streamlit / Gradio web UI
- [ ] PDF & DOCX document support
- [ ] Multi-query retrieval with query expansion
- [ ] Reranking with Cross-Encoder models
- [ ] Evaluation metrics (MRR, NDCG, faithfulness)

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

```bash
# Fork → Clone → Branch → Commit → Push → PR
git checkout -b feature/your-feature-name
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
```

---

<div align="center">

**Built with ❤️ to learn RAG from the ground up**

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

</div>
