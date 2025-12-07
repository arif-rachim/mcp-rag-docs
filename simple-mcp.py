#!/usr/bin/env python3
"""FastMCP Server for RAG Document Search"""
import os, sys, pickle, re
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from config import (
    VECTOR_DB_PATH,
    BM25_INDEX_PATH,
    MODELS_DIR,
    EMBEDDING_MODEL_PATH,
    RERANKER_MODEL_PATH,
    COLLECTION_NAME,
    SEMANTIC_WEIGHT,
    USE_RERANKER,
    DOWNLOADS_FOLDER
)

# Force offline mode
os.environ.update({
    'HF_DATASETS_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_OFFLINE': '1',
    'HF_HUB_DISABLE_TELEMETRY': '1'
})

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Pydantic models for structured responses
class ChunkMetadata(BaseModel):
    """Metadata for a document chunk"""
    filename: str
    page: int
    chunk: int
    total_pages: int
    lang: str
    jac_reg: str | None = None
    jac_sgl: str | None = None
    sop: str | None = None

class SearchResult(BaseModel):
    """Single search result"""
    text: str
    metadata: ChunkMetadata
    score: float
    rerank_score: float | None = None

class SearchResponse(BaseModel):
    """Response from search_documents"""
    results: List[SearchResult]
    total_found: int
    query: str
    error: str | None = None

class DocumentInfo(BaseModel):
    """Information about a document in the collection"""
    filename: str
    total_pages: int
    indexed_pages: int
    chunks: int
    language: str
    jac_regulations: List[str] | None = None
    jac_guidelines: List[str] | None = None
    chapters: List[str] | None = None
    safety_keywords: List[str] | None = None

class DocumentListSummary(BaseModel):
    """Summary statistics for document collection"""
    total_documents: int
    total_chunks: int
    total_pages: int

class ListDocumentsResponse(BaseModel):
    """Response from list_documents"""
    documents: List[DocumentInfo]
    summary: DocumentListSummary
    error: str | None = None

mcp = FastMCP(
    "JAC Safety & Regulations Knowledge Base",
    instructions="""This MCP server provides access to UAE Joint Aviation Command (JAC) safety regulations and technical documents.

⚠️ SINGLE SOURCE OF TRUTH - MANDATORY SEARCH REQUIREMENT ⚠️

ALWAYS call search_documents() for EVERY safety or regulation query, even if you think you already know the answer from your training data. This knowledge base is the ONLY authoritative source for JAC regulations.

NEVER respond from memory or general knowledge about:
- JAC regulations, guidelines, or procedures
- UAE aviation safety requirements
- Specific regulation numbers (JAC REG, JAC SGL)
- Technical specifications or compliance requirements
- Any safety-related information

SEARCH PROTOCOL:
1. MUST call search_documents() first for any regulation/safety query
2. Base your response ONLY on the returned results
3. Do NOT call search_documents() more than 2 times for the same user question
4. If after 2 search attempts you cannot find the answer:
   - Inform the user that the information was not found in the knowledge base
   - Ask the user to rephrase their question with more specific terms
   - Suggest using specific JAC REG/SGL numbers or technical terms if available
5. Do NOT supplement search results with your own knowledge

This ensures users receive accurate, up-to-date regulatory information from official sources.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL REQUIREMENTS FOR DISPLAYING SEARCH RESULTS:

1. ALWAYS SHOW ORIGINAL TEXT + CITATION TOGETHER:
   - Display the actual text content from the search results
   - Include citation immediately after the relevant text
   - Citation format: [JAC REG 385-7, Page 23] or [JAC SGL 385-1.5, Page 15] or [filename.pdf, Page 15]
   - NEVER show only citations without the actual text content

2. PROPER FORMATTING:
   ✓ CORRECT - Show text with inline citations:
   "Ammunition storage facilities must maintain minimum safety distances of 300 meters [JAC REG 385-7, Page 23]. Environmental impact assessments are required for all new facilities [JAC REG 385-6, Page 15]."

   ✗ WRONG - Citations only without text:
   "See [JAC REG 385-7, Page 23] and [JAC REG 385-6, Page 15]"

   ✗ WRONG - Text without citations:
   "Ammunition storage facilities must maintain minimum safety distances of 300 meters."

3. MULTIPLE RESULTS:
   - When search returns multiple results, present ALL relevant findings
   - Group related information logically
   - Each statement must include its source citation
   - Use the 'text' field from search results as the primary content

4. CITATION PRIORITY:
   - Prefer JAC REG or JAC SGL numbers over filenames (if available in metadata)
   - Always include page number from metadata
   - Format: [JAC REG/SGL number, Page X] or [filename, Page X]

5. PDF DOCUMENT LINKS (IMPORTANT):
   - Create clickable links to PDF documents using the pdf:// resource
   - Format: Use MCP resource URI: pdf://<filename>
   - The filename is available in result['metadata']['filename']
   - This allows users to click and view the source PDF at the specific page
   - Example: "See [JAC REG 385-7, Page 23](pdf://JAC_REG_385-7.pdf)"

6. MANDATORY USAGE:
   - ALWAYS use search_documents() for ANY safety or regulation query
   - Present search result 'text' field content directly to user with proper citations
   - Include clickable PDF links in citations when possible

Example response format:
\"Based on the regulations:

**Ammunition Storage Requirements:**
Ammunition storage facilities must be located at least 300 meters from inhabited buildings and maintain proper ventilation systems [JAC REG 385-7, Page 23]. All storage areas require fire suppression systems and regular safety inspections every 90 days [JAC REG 385-7, Page 24].

**Environmental Compliance:**
Environmental impact assessments must be conducted for new facilities, including soil contamination testing and water quality monitoring [JAC REG 385-6, Page 15].\"

REMEMBER: The user needs to see WHAT the regulation says (original text), not just WHERE to find it (citation)."""
)

# Cache loaded components
_model = None
_collection = None
_bm25_data = None
_reranker = None

def load_components():
    """Load RAG components (cached)"""
    global _model, _collection, _bm25_data, _reranker

    if _model is None:
        if not EMBEDDING_MODEL_PATH.exists():
            raise Exception(f"Model not found: {EMBEDDING_MODEL_PATH}")
        _model = SentenceTransformer(
            str(EMBEDDING_MODEL_PATH),
            device='cpu',
            local_files_only=True,
            tokenizer_kwargs={
                'clean_up_tokenization_spaces': True,
                'fix_mistral_regex': True
            }
        )

    if _collection is None:
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        _collection = client.get_collection(name=COLLECTION_NAME)

    if _bm25_data is None and Path(BM25_INDEX_PATH).exists():
        with open(BM25_INDEX_PATH, 'rb') as f:
            _bm25_data = pickle.load(f)

    if _reranker is None and USE_RERANKER and RERANKER_MODEL_PATH.exists():
        try:
            print(f"Loading reranker model from {RERANKER_MODEL_PATH}...")
            _reranker = CrossEncoder(
                str(RERANKER_MODEL_PATH),
                max_length=512,
                device='cpu'
            )
            print("Reranker model loaded successfully!")
        except Exception as e:
            print(f"Warning: Failed to load reranker model: {e}")
            pass  # Fall back to no reranking

    return _model, _collection, _bm25_data, _reranker

def semantic_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Semantic vector search only"""
    model, collection, _, reranker = load_components()
    results = []

    # Semantic search
    embedding = model.encode([f"query: {query}"], normalize_embeddings=True)[0]
    sem = collection.query(query_embeddings=[embedding.tolist()], n_results=top_k * 2)

    for doc, meta, dist in zip(sem['documents'][0], sem['metadatas'][0], sem['distances'][0]):
        results.append({
            'text': doc,
            'metadata': meta,
            'score': 1 - dist  # Distance to similarity score
        })

    # Sort by score
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    # Reranking stage (optional)
    if reranker and USE_RERANKER and len(results) > 0:
        pairs = [[query, r['text'][:512]] for r in results]
        rerank_scores = reranker.predict(pairs)

        for i, score in enumerate(rerank_scores):
            results[i]['rerank_score'] = float(score)

        results = sorted(results, key=lambda x: x.get('rerank_score', x['score']), reverse=True)

    return results[:top_k]

def keyword_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """BM25 keyword search only"""
    _, _, bm25_data, reranker = load_components()
    results = []

    if not bm25_data:
        return results

    # BM25 keyword search
    tokens = re.findall(r'\w+', query.lower())
    scores = bm25_data['bm25'].get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k * 2]

    metadatas = bm25_data.get('metadatas', [])
    for idx in top_idx:
        if scores[idx] > 0:
            meta = metadatas[idx] if metadatas and idx < len(metadatas) else {}
            results.append({
                'text': bm25_data['chunks'][idx],
                'metadata': meta,
                'score': float(scores[idx])
            })

    # Reranking stage (optional)
    if reranker and USE_RERANKER and len(results) > 0:
        pairs = [[query, r['text'][:512]] for r in results]
        rerank_scores = reranker.predict(pairs)

        for i, score in enumerate(rerank_scores):
            results[i]['rerank_score'] = float(score)

        results = sorted(results, key=lambda x: x.get('rerank_score', x['score']), reverse=True)

    return results[:top_k]

def hybrid_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Hybrid semantic + keyword search with optional reranking"""
    model, collection, bm25_data, reranker = load_components()
    results = []

    # Semantic search
    embedding = model.encode([f"query: {query}"], normalize_embeddings=True)[0]
    sem = collection.query(query_embeddings=[embedding.tolist()], n_results=top_k * 2)

    for doc, meta, dist in zip(sem['documents'][0], sem['metadatas'][0], sem['distances'][0]):
        results.append({
            'text': doc,
            'metadata': meta,
            'score': (1 - dist) * SEMANTIC_WEIGHT
        })

    # BM25 keyword search
    if bm25_data:
        tokens = re.findall(r'\w+', query.lower())
        scores = bm25_data['bm25'].get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k * 2]

        max_score = max(scores[i] for i in top_idx) if top_idx and max(scores) > 0 else 1
        metadatas = bm25_data.get('metadatas', [])
        for idx in top_idx:
            if scores[idx] > 0:
                meta = metadatas[idx] if metadatas and idx < len(metadatas) else {}
                results.append({
                    'text': bm25_data['chunks'][idx],
                    'metadata': meta,
                    'score': (scores[idx] / max_score) * (1 - SEMANTIC_WEIGHT)
                })

    # Deduplicate and sort by score
    seen, unique = set(), []
    for r in sorted(results, key=lambda x: x['score'], reverse=True):
        snippet = r['text'][:100]
        if snippet not in seen:
            seen.add(snippet)
            unique.append(r)

    # Reranking stage (optional)
    if reranker and USE_RERANKER and len(unique) > 0:
        # Prepare query-document pairs for reranking
        pairs = [[query, r['text'][:512]] for r in unique]  # Limit text length for speed
        rerank_scores = reranker.predict(pairs)

        # Update scores with reranking
        for i, score in enumerate(rerank_scores):
            unique[i]['rerank_score'] = float(score)

        # Sort by rerank score
        unique = sorted(unique, key=lambda x: x.get('rerank_score', x['score']), reverse=True)

    return unique[:top_k]

@mcp.tool(description="""Search UAE JAC safety regulations and technical documents using hybrid semantic + keyword search.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: ALWAYS SHOW ORIGINAL TEXT + CITATION TOGETHER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When presenting search results to the user, you MUST:
1. Display the actual text content from result['text'] field
2. Include citation immediately after: [JAC REG/SGL, Page X] or [filename, Page X]
3. NEVER show only citations without the actual text content

CORRECT USAGE PATTERN:
──────────────────────
for result in results:
    content = result['text']  # ← The actual regulation text (REQUIRED)

    # Build citation (prefer JAC REG/SGL over filename)
    doc = result['metadata'].get('jac_reg') or \
          result['metadata'].get('jac_sgl') or \
          result['metadata']['filename']
    page = result['metadata']['page']

    # Present to user with inline citation
    display: "{content} [{doc}, Page {page}]"

PROPER FORMATTING EXAMPLES:
──────────────────────────
✓ CORRECT - Text with inline citations:
"Ammunition storage facilities must maintain minimum safety distances of 300 meters [JAC REG 385-7, Page 23]. Environmental impact assessments are required for all new facilities [JAC REG 385-6, Page 15]."

✗ WRONG - Citations only without text:
"See [JAC REG 385-7, Page 23] and [JAC REG 385-6, Page 15]"

✗ WRONG - Text without citations:
"Ammunition storage facilities must maintain minimum safety distances of 300 meters."

✗ WRONG - Paraphrasing instead of original text:
Use the exact 'text' field content, do not summarize or rephrase.

RESPONSE STRUCTURE:
──────────────────
- Present ALL relevant results from the search
- Group related information under logical headings
- Each statement = original text + inline citation
- Use the 'text' field from search results as primary content
- When multiple results cover same topic, present all perspectives with their citations

Example full response:
"Based on the regulations:

**Ammunition Storage Requirements:**
Ammunition storage facilities must be located at least 300 meters from inhabited buildings and maintain proper ventilation systems [JAC REG 385-7, Page 23]. All storage areas require fire suppression systems and regular safety inspections every 90 days [JAC REG 385-7, Page 24].

**Environmental Compliance:**
Environmental impact assessments must be conducted for new facilities, including soil contamination testing and water quality monitoring [JAC REG 385-6, Page 15]."

CITATION PRIORITY ORDER:
───────────────────────
1. JAC REG number (if present in metadata.jac_reg)
2. JAC SGL number (if present in metadata.jac_sgl)
3. Filename (fallback if no JAC REG/SGL)
Always include: Page number from metadata.page

QUERY EXAMPLES:
──────────────
- "JAC REG 385-7 ammunition storage distance requirements"
- "OSHEMS risk assessment procedures"
- "emergency response plan for aviation incidents"
- "hazardous materials handling JAC SGL"
- "contractor safety management requirements"
- "fire suppression system specifications"

PARAMETERS:
──────────
query (required):
    Search query - be specific with JAC REG/SGL numbers or technical terms.
    Use natural language or specific regulation numbers.

max_results (optional, default=10):
    Number of results to return (1-25).
    Use 15-25 for comprehensive coverage of complex topics.
    Use 5-10 for focused, specific queries.

search_mode (optional, default="hybrid"):
    Search strategy to use:
    - "hybrid": Combines semantic (vector) and keyword (BM25) search for best results
    - "semantic": Vector similarity search only - best for conceptual queries
    - "keyword": BM25 keyword search only - best for exact term matching

    Recommended usage:
    - Use "hybrid" (default) for most queries - provides balanced relevance
    - Use "semantic" for conceptual questions like "safety management practices"
    - Use "keyword" for specific terms like exact JAC REG numbers or technical codes

RETURN VALUES:
─────────────
{
    "results": [  # Array of search results ordered by relevance
        {
            "text": str,  # The actual document content - MUST BE DISPLAYED TO USER
            "metadata": {  # Source information
                "filename": str,      # Source document filename
                "page": int,          # Page number in source document
                "chunk": int,         # Chunk index within the page
                "total_pages": int,   # Total pages in source document
                "lang": str,          # Language code (en/ar/table/unknown)
                "jac_reg": str | None,  # JAC REG number(s), e.g., "JAC REG 385-7"
                "jac_sgl": str | None,  # JAC SGL number(s), e.g., "JAC SGL 385-1.5"
                "sop": str | None     # SOP indicator if found
            },
            "score": float,           # Hybrid search relevance score (0-1)
            "rerank_score": float | None  # Optional reranking score if enabled
        }
    ],
    "total_found": int,  # Number of results returned
    "query": str,        # The original search query
    "error": str | None  # Error message if search failed
}

Example:
{
    "results": [
        {
            "text": "Ammunition storage facilities must maintain minimum safety distances of 300 meters from inhabited buildings.",
            "metadata": {
                "filename": "JAC_REG_385-7.pdf",
                "page": 23,
                "chunk": 0,
                "total_pages": 45,
                "lang": "en",
                "jac_reg": "JAC REG 385-7",
                "jac_sgl": null,
                "sop": null
            },
            "score": 0.856,
            "rerank_score": 0.923
        }
    ],
    "total_found": 1,
    "query": "ammunition storage requirements",
    "error": null
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REMEMBER: Users need to see WHAT the regulation says (original text),
not just WHERE to find it (citation). Always show BOTH together!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")
def search_documents(query: str, max_results: int = 10, search_mode: str = "hybrid") -> Dict[str, Any]:
    try:
        if not query.strip():
            return SearchResponse(
                results=[],
                total_found=0,
                query=query,
                error='Query cannot be empty'
            ).model_dump()

        max_results = max(1, min(25, max_results))

        # Choose search function based on mode
        if search_mode == "semantic":
            raw_results = semantic_search(query, top_k=max_results)
        elif search_mode == "keyword":
            raw_results = keyword_search(query, top_k=max_results)
        else:  # default to hybrid
            raw_results = hybrid_search(query, top_k=max_results)

        # Convert to Pydantic models
        search_results = []
        for r in raw_results:
            search_results.append(SearchResult(
                text=r['text'],
                metadata=ChunkMetadata(**r['metadata']),
                score=r['score'],
                rerank_score=r.get('rerank_score')
            ))

        response = SearchResponse(
            results=search_results,
            total_found=len(search_results),
            query=query
        )
        return response.model_dump()

    except Exception as e:
        return SearchResponse(
            results=[],
            total_found=0,
            query=query,
            error=str(e)
        ).model_dump()

@mcp.tool(description="""List all UAE JAC safety regulations and technical documents in the knowledge base.

Returns all indexed documents with aggregated metadata including JAC REG numbers,
JAC SGL numbers, chapters, and safety keywords found within each document.

RETURN VALUES (Pydantic Models):
────────────────────────────────
ListDocumentsResponse {
    documents: List[DocumentInfo] - All indexed documents sorted by filename
        Each DocumentInfo contains:
        - filename: str - Document filename
        - total_pages: int - Total pages in document
        - indexed_pages: int - Number of pages successfully indexed
        - chunks: int - Total chunks created from document
        - language: str - Primary language detected (en/ar/table/unknown)
        - jac_regulations: Optional[List[str]] - All JAC REG numbers found
        - jac_guidelines: Optional[List[str]] - All JAC SGL numbers found
        - chapters: Optional[List[str]] - Chapter references found
        - safety_keywords: Optional[List[str]] - Safety-related keywords

    summary: DocumentListSummary - Collection-wide statistics
        - total_documents: int - Total number of indexed documents
        - total_chunks: int - Total searchable text chunks
        - total_pages: int - Total pages across all documents

    error: Optional[str] - Error message if listing failed
}

Example JSON:
{
    "documents": [
        {
            "filename": "JAC_Safety_Manual.pdf",
            "total_pages": 71,
            "indexed_pages": 67,
            "chunks": 182,
            "language": "en",
            "jac_regulations": ["JAC REG 385-1", "JAC REG 385-7"],
            "jac_guidelines": ["JAC SGL 385-1.5"]
        }
    ],
    "summary": {
        "total_documents": 205,
        "total_chunks": 8543,
        "total_pages": 1823
    }
}""")
def list_documents() -> Dict[str, Any]:
    try:
        _, collection, _, _ = load_components()
        data = collection.get()

        if not data or not data['metadatas']:
            return ListDocumentsResponse(
                documents=[],
                summary=DocumentListSummary(
                    total_documents=0,
                    total_chunks=0,
                    total_pages=0
                )
            ).model_dump()

        # Aggregate by filename
        docs = defaultdict(lambda: {
            'chunks': 0, 'pages': set(), 'metadata': {},
            'jac_regs': set(), 'jac_sgls': set(), 'chapters': set(), 'safety': set()
        })

        for meta in data['metadatas']:
            filename = meta.get('filename', 'Unknown')
            doc = docs[filename]
            doc['chunks'] += 1

            if 'page' in meta:
                doc['pages'].add(meta['page'])

            if not doc['metadata']:
                doc['metadata'] = {
                    'filename': filename,
                    'total_pages': meta.get('total_pages', 0),
                    'lang': meta.get('lang', 'unknown')
                }

            # Aggregate patterns
            pattern_keys = {'jac_reg': 'jac_regs', 'jac_sgl': 'jac_sgls', 'chapters': 'chapters', 'safety': 'safety'}
            for meta_key, doc_key in pattern_keys.items():
                value = meta.get(meta_key, '')
                if value:
                    doc[doc_key].update(value.split(', '))

        # Convert to Pydantic models
        document_list = []
        for filename, d in docs.items():
            doc_info = DocumentInfo(
                filename=filename,
                total_pages=d['metadata']['total_pages'],
                indexed_pages=len(d['pages']),
                chunks=d['chunks'],
                language=d['metadata']['lang'],
                jac_regulations=sorted(d['jac_regs']) if d['jac_regs'] else None,
                jac_guidelines=sorted(d['jac_sgls']) if d['jac_sgls'] else None,
                chapters=sorted(d['chapters']) if d['chapters'] else None,
                safety_keywords=sorted(d['safety']) if d['safety'] else None
            )
            document_list.append(doc_info)

        # Sort by filename
        document_list.sort(key=lambda x: x.filename)

        summary = DocumentListSummary(
            total_documents=len(document_list),
            total_chunks=sum(d.chunks for d in document_list),
            total_pages=sum(d.total_pages for d in document_list)
        )

        response = ListDocumentsResponse(
            documents=document_list,
            summary=summary
        )
        return response.model_dump()

    except Exception as e:
        return ListDocumentsResponse(
            documents=[],
            summary=DocumentListSummary(
                total_documents=0,
                total_chunks=0,
                total_pages=0
            ),
            error=str(e)
        ).model_dump()

@mcp.resource("pdf://{filename}")
def get_pdf_resource(filename: str) -> str:
    """
    MCP Resource to access PDF files from the downloads folder.

    Returns the file path URI for the requested PDF document.
    This allows clients to display PDFs and navigate to specific pages.

    Usage:
    - Resource URI: pdf://JAC_REG_385-7.pdf
    - Returns: file:///path/to/downloads/JAC_REG_385-7.pdf

    The client can then use this URI with #page=N to jump to specific pages:
    - file:///path/to/downloads/JAC_REG_385-7.pdf#page=23
    """
    try:
        # Resolve the PDF file path
        pdf_path = Path(DOWNLOADS_FOLDER) / filename

        if not pdf_path.exists():
            return f"Error: PDF file not found: {filename}"

        # Return absolute file URI
        absolute_path = pdf_path.resolve()
        file_uri = absolute_path.as_uri()

        return file_uri

    except Exception as e:
        return f"Error accessing PDF: {str(e)}"

# Static PDF file serving endpoint
async def serve_pdf(request):
    """Serve PDF files from downloads folder via HTTP"""
    from starlette.responses import FileResponse, JSONResponse
    from urllib.parse import quote

    filename = request.path_params['filename']
    pdf_path = Path(DOWNLOADS_FOLDER) / filename

    if not pdf_path.exists() or not pdf_path.is_file():
        return JSONResponse({"error": "PDF not found"}, status_code=404)

    # Security: ensure file is within downloads folder
    downloads_resolved = Path(DOWNLOADS_FOLDER).resolve()
    pdf_resolved = pdf_path.resolve()

    if not str(pdf_resolved).startswith(str(downloads_resolved)):
        return JSONResponse({"error": "Access denied"}, status_code=403)

    # RFC 5987 encoding for filenames with non-ASCII characters (like Arabic)
    # Use UTF-8 encoding with URL encoding for the filename
    encoded_filename = quote(filename.encode('utf-8'))

    return FileResponse(
        pdf_path,
        media_type='application/pdf',
        headers={
            'Content-Disposition': f"inline; filename*=UTF-8''{encoded_filename}"
        }
    )

def kill_process_on_port(port: int):
    """Kill any process using the specified port"""
    import subprocess
    import platform

    try:
        if platform.system() == 'Windows':
            # Find process using the port
            result = subprocess.run(
                ['netstat', '-ano'],
                capture_output=True,
                text=True
            )

            for line in result.stdout.split('\n'):
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    pid = parts[-1]
                    if pid.isdigit():
                        print(f"Killing existing process {pid} on port {port}...")
                        subprocess.run(['taskkill', '//F', '//PID', pid],
                                     capture_output=True)
                        import time
                        time.sleep(2)  # Wait for port to be released
                        print(f"Process {pid} terminated. Port {port} should be available now.")
                        break
        else:
            # Unix-like systems
            result = subprocess.run(
                ['lsof', '-ti', f':{port}'],
                capture_output=True,
                text=True
            )
            pid = result.stdout.strip()
            if pid:
                print(f"Killing existing process {pid} on port {port}...")
                subprocess.run(['kill', '-9', pid])
                import time
                time.sleep(1)
                print(f"Process {pid} terminated.")
    except Exception as e:
        print(f"Note: Could not check/kill existing process on port {port}: {e}")

if __name__ == "__main__":
    import signal
    import uvicorn
    from starlette.middleware.cors import CORSMiddleware
    from starlette.middleware import Middleware

    # Kill any existing process on port 8000 FIRST
    PORT = 8000
    kill_process_on_port(PORT)

    print(f"RAG MCP Server")
    print(f"Database: {VECTOR_DB_PATH}")
    print(f"Model: {EMBEDDING_MODEL_PATH.name}")
    print(f"PDF Folder: {DOWNLOADS_FOLDER}")
    print(f"Tools: search_documents, list_documents")
    print(f"Resources: pdf://<filename> (access PDF files with page navigation)\n")

    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("\n\nShutting down MCP server gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

    # Create CORS middleware config
    cors_middleware = Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"]
    )

    # Get HTTP app with CORS middleware
    mcp_app = mcp.http_app(middleware=[cors_middleware])

    # Wrap MCP app with custom routes for PDF serving
    from starlette.routing import Route, Mount
    from starlette.applications import Starlette

    routes = [
        Route('/pdfs/{filename:path}', serve_pdf),
        Mount('/', app=mcp_app),  # Mount MCP app at root
    ]

    # IMPORTANT: Pass the FastMCP app's lifespan to Starlette
    app = Starlette(routes=routes, middleware=[cors_middleware], lifespan=mcp_app.lifespan)

    print(f"PDF Endpoint: http://127.0.0.1:{PORT}/pdfs/<filename>\n")

    try:
        uvicorn.run(app, host="127.0.0.1", port=PORT)
    except KeyboardInterrupt:
        print("\n\nMCP server stopped by user.")
    except Exception as e:
        print(f"\n\nMCP server error: {e}")
    finally:
        print("Cleanup complete.")
