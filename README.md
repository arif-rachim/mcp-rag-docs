# MCP RAG Docs

MCP RAG Docs is a local retrieval-augmented generation (RAG) system for searching collections of PDF documents such as technical manuals and safety regulations, in English and Arabic. It exists so that an LLM client can answer questions from a private document set without sending the documents anywhere: search is exposed as an MCP server, so any MCP-capable client can call it as a tool and quote the original text with page citations. An ingest script extracts text and tables from the PDFs with PyMuPDF, splits them into chunks of about 800 characters, and builds a ChromaDB vector store (multilingual-e5-large embeddings) plus a BM25 keyword index. The FastMCP server combines both scores and can rerank results with a cross-encoder, and a React web UI lets people search and read the PDFs by hand. Everything runs offline once the models have been downloaded. It is a working set of scripts rather than a packaged product.

## Features

- Offline operation: models are downloaded once into `./models` and loaded from disk with Hugging Face offline mode forced on
- PDF ingestion with PyMuPDF, including table extraction and simple English/Arabic language detection
- Paragraph-based chunking (about 800 characters, 100-character overlap cut at word boundaries)
- Optional named-entity extraction with `Davlan/bert-base-multilingual-cased-ner-hrl` when that model is present
- Parallel ingestion with `multiprocessing`; each run rebuilds the indexes from scratch
- Hybrid search: 70% semantic score + 30% normalised BM25 score, deduplicated, then optionally reranked with `BAAI/bge-reranker-v2-m3`
- Three search modes: `hybrid`, `semantic` and `keyword`, with up to 25 results per query and matched terms returned for highlighting
- PDFs available as MCP resources (`pdf://<filename>`) and over HTTP (`/pdfs/<filename>`)
- React web UI with a search box, highlighted results, a document list, a connection status indicator and a built-in PDF viewer
- Optional SharePoint downloader for fetching the PDFs from a document library

## Tech stack

Python · FastMCP · ChromaDB · sentence-transformers · rank_bm25 · PyMuPDF · Hugging Face Transformers · Starlette/uvicorn · React 19 · Vite · Tailwind CSS 4 · react-pdf · use-mcp

## How it works

1. **Ingest** (`simple-rag.py`): reads the PDFs in `./downloads`, splits them into chunks, extracts metadata, and builds:
   - a **ChromaDB** vector store (`./chroma_store`, collection `technical_documents`) using `intfloat/multilingual-e5-large` embeddings
   - a **BM25** keyword index (`./bm25_index.pkl`)
2. **Search** (`simple-mcp.py`): a FastMCP server over HTTP on `127.0.0.1:8000` (MCP endpoint `/mcp`, CORS open). It runs the hybrid search and exposes:
   - `search_documents(query, max_results, search_mode)`
   - `list_documents()`
   - the PDFs as resources (`pdf://<filename>`) and as files at `/pdfs/<filename>`

   The tool descriptions instruct the calling LLM to show the original text followed by a citation such as `[JAC REG/SGL number, Page X]` or `[filename, Page X]`; they are written for a set of safety regulation documents and can be adapted to other collections. On start-up the server kills any process already listening on port 8000.
3. **Web UI** (`simple-webui/`): a React, Vite and Tailwind client that talks to the MCP server with `use-mcp` and shows PDFs with `react-pdf`.

## Getting started

Prerequisites: Python 3 and Node.js.

```bash
pip install chromadb sentence-transformers rank_bm25 fastmcp pymupdf transformers tqdm uvicorn

python simple-model-downloader.py   # download embedding, NER and reranker models into ./models (one-time)
# put PDFs into ./downloads  (or use simple-downloader.py to pull them from SharePoint)
python simple-rag.py                # build the vector and BM25 indexes (add --sequential to disable multiprocessing)
python simple-mcp.py                # start the MCP server on http://127.0.0.1:8000/mcp
```

`simple-downloader.py` additionally needs `requests` and `requests_ntlm`. Its site URL, library name and NTLM credentials are set as constants at the top of the file.

Web UI:

```bash
cd simple-webui
npm install
npm run dev       # development server
npm run build     # production build
npm run lint
```

Environment variable for the web UI: `VITE_MCP_SERVER_URL` (defaults to `http://localhost:8000/mcp`).

## Configuration

Settings shared by the scripts are in `config.py`: model IDs and local model paths, the vector store and BM25 index paths, the collection name, the downloads folder, `SEMANTIC_WEIGHT` (default `0.7`) and `USE_RERANKER` (default `True`). Chunk size and overlap are set in `simple-rag.py`.

## Project structure

```text
config.py                     shared paths, model IDs and search settings
simple-model-downloader.py    downloads the models for offline use
simple-downloader.py          optional SharePoint PDF downloader (NTLM)
simple-rag.py                 ingestion: PDF extraction, chunking, embeddings, BM25
simple-mcp.py                 FastMCP server: hybrid search, tools, PDF resources
test_highlights.py            checks keyword highlighting for English and Arabic queries
simple-webui/                 React + Vite search UI and PDF viewer
```

## Limitations

- The server binds to `127.0.0.1:8000`, allows all CORS origins and has no authentication; it is meant for local use.
- Re-running ingestion deletes and rebuilds the existing indexes.
