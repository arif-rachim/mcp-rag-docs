# MCP RAG Docs

A local RAG (retrieval-augmented generation) system for searching PDF documents, such as technical documents and safety regulations. Search is exposed as an **MCP server**, so an LLM client can call it as a tool. A small React web UI is included for searching and reading documents by hand.

Everything runs **offline**. The models are downloaded once and then loaded from local disk.

## How it works

1. **Ingest** (`simple-rag.py`): reads the PDFs in `./downloads`, splits them into chunks of about 800 characters with some overlap, pulls out metadata, and builds:
   - a **ChromaDB** vector store using `intfloat/multilingual-e5-large` embeddings
   - a **BM25** keyword index
2. **Search** (`simple-mcp.py`): a FastMCP server that runs hybrid search. The final score is 70% semantic and 30% BM25, and results can be reranked with `BAAI/bge-reranker-v2-m3`. It exposes these tools:
   - `search_documents(query, max_results, search_mode)`
   - `list_documents()`
   - access to the PDFs as resources
3. **Web UI** (`simple-webui/`): a React, Vite and Tailwind client. It has a search box, highlighted results and a built-in PDF viewer.

## Setup

```bash
pip install chromadb sentence-transformers rank_bm25 fastmcp pymupdf transformers tqdm

python simple-model-downloader.py   # download models into ./models (one-time)
# put PDFs into ./downloads  (or use simple-downloader.py to pull from SharePoint)
python simple-rag.py                # build vector + BM25 indexes
python simple-mcp.py                # start the MCP server
```

Web UI:

```bash
cd simple-webui
npm install
npm run dev     # set VITE_MCP_SERVER_URL in .env
```

## Configuration

Settings shared by the scripts are in `config.py`: model paths, index paths, the collection name, `SEMANTIC_WEIGHT` and `USE_RERANKER`.
