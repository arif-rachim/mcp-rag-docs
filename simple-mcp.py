#!/usr/bin/env python3
"""
FastMCP Server for RAG System
Exposes technical document search capabilities via Model Context Protocol
"""

# ============================================================================
# SET ENVIRONMENT VARIABLES (MUST BE BEFORE IMPORTS!)
# ============================================================================
import os
import sys
from pathlib import Path

# RAG Configuration (matching simple-rag.py)
VECTOR_DB_PATH = './database'
BM25_INDEX_PATH = './bm25_index.pkl'
MODELS_DIR = './models'
EMBEDDING_MODEL = 'intfloat/multilingual-e5-large'
COLLECTION_NAME = 'technical_documents'
SEMANTIC_WEIGHT = 0.7
USE_LOCAL_MODELS_ONLY = True

# Set environment for offline mode - MUST BE BEFORE OTHER IMPORTS
os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(Path(MODELS_DIR).absolute())
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'  # Prevents HuggingFace Hub API calls
os.environ['HUGGINGFACE_HUB_CACHE'] = str(Path(MODELS_DIR).absolute())

# ============================================================================
# IMPORTS
# ============================================================================
import pickle
from collections import defaultdict
from typing import List, Dict, Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from fastmcp import FastMCP

# ============================================================================
# CONFIGURATION
# ============================================================================

# Initialize FastMCP
mcp = FastMCP("RAG Document Search Server")

# ============================================================================
# RAG HELPER FUNCTIONS
# ============================================================================

def load_rag_components():
    """
    Load embedding model, ChromaDB collection, and BM25 index.
    Returns: (model, collection, bm25_data) tuple
    Raises: Exception if components cannot be loaded
    """
    try:
        # Load ChromaDB
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)

        # Load embedding model from local path
        models_path = Path(MODELS_DIR).absolute()

        # Construct the local model path (simple structure: models/model-name/)
        model_name = EMBEDDING_MODEL.split('/')[-1]  # Get just "multilingual-e5-large"
        embedding_model_path = models_path / model_name

        # Check if model exists locally
        if not embedding_model_path.exists():
            raise Exception(f"Model not found at: {embedding_model_path}. Run: python simple-model-downloader.py")

        # Load from the actual local path (not model name)
        model = SentenceTransformer(
            str(embedding_model_path),  # Use local path instead of model name
            device='cpu'  # Force CPU to avoid CUDA issues
        )

        # Load BM25 index
        bm25_data = None
        if os.path.exists(BM25_INDEX_PATH):
            with open(BM25_INDEX_PATH, 'rb') as f:
                bm25_data = pickle.load(f)

        return model, collection, bm25_data

    except Exception as e:
        raise Exception(f"Failed to load RAG components: {str(e)}")


def perform_hybrid_search(
    query: str,
    model,
    collection,
    bm25_data,
    top_k: int = 10,
    alpha: float = SEMANTIC_WEIGHT
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining semantic (E5) + keyword (BM25).

    This is adapted from simple-rag.py's hybrid_search() function.

    Args:
        query: Search query string
        model: SentenceTransformer model
        collection: ChromaDB collection
        bm25_data: BM25 index data dict
        top_k: Maximum number of results to return
        alpha: Weight for semantic search (1-alpha is BM25 weight)

    Returns:
        List of dicts with keys: 'text', 'metadata', 'score'
    """
    results = []

    # Semantic search
    query_embedding = model.encode([f"query: {query}"])[0]
    sem_results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k * 2
    )

    for doc, meta, dist in zip(
        sem_results['documents'][0],
        sem_results['metadatas'][0],
        sem_results['distances'][0]
    ):
        results.append({
            'text': doc,
            'metadata': meta,
            'score': (1 - dist) * alpha
        })

    # BM25 search
    if bm25_data:
        scores = bm25_data['bm25'].get_scores(query.lower().split())
        top_idx = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k * 2]

        max_score = max(scores[i] for i in top_idx) if top_idx else 1
        if max_score > 0:
            for idx in top_idx:
                results.append({
                    'text': bm25_data['chunks'][idx],
                    'metadata': {},
                    'score': (scores[idx] / max_score) * (1 - alpha)
                })

    # Deduplicate and sort
    seen = set()
    unique = []
    for r in sorted(results, key=lambda x: x['score'], reverse=True):
        snippet = r['text'][:100]
        if snippet not in seen:
            seen.add(snippet)
            unique.append(r)

    return unique[:top_k]


def aggregate_documents(collection) -> List[Dict[str, Any]]:
    """
    Aggregate document-level information from chunk-level ChromaDB data.

    Since ChromaDB stores chunks, we need to aggregate metadata by filename
    to provide a document-level view.

    Returns:
        List of document dicts with aggregated metadata
    """
    try:
        # Fetch all data from collection
        all_data = collection.get()

        if not all_data or not all_data['metadatas']:
            return []

        # Aggregate by filename
        docs_by_filename = defaultdict(lambda: {
            'chunks': [],
            'metadata': {},
            'pages': set(),
            'jac_regs': set(),
            'chapters': set(),
            'safety_keywords': set(),
            'acronyms': set(),
            'sms_terms': set(),
        })

        for meta in all_data['metadatas']:
            filename = meta.get('filename', 'Unknown')
            doc = docs_by_filename[filename]

            # Count chunks
            doc['chunks'].append(1)

            # Collect unique pages
            if 'page_number' in meta:
                doc['pages'].add(meta['page_number'])

            # Store base metadata (from first chunk)
            if not doc['metadata']:
                doc['metadata'] = {
                    'filename': filename,
                    'doc_type': meta.get('doc_type', 'document'),
                    'pdf_title': meta.get('pdf_title', filename),
                    'total_pages': meta.get('total_pages', len(doc['pages'])),
                    'classification': meta.get('classification', ''),
                }

            # Aggregate pattern-based metadata
            for key in ['jac_reg_numbers', 'chapters', 'safety_keywords', 'acronyms', 'sms_terms']:
                if key in meta and meta[key]:
                    # Split comma-separated values and add to set
                    values = [v.strip() for v in meta[key].split(',')]
                    short_key = key.replace('_numbers', 's').replace('_keywords', 's').replace('_terms', 's')
                    doc[short_key].update(values)

        # Format final output
        documents = []
        for filename, data in docs_by_filename.items():
            doc_info = {
                'filename': data['metadata']['filename'],
                'pdf_title': data['metadata']['pdf_title'],
                'doc_type': data['metadata']['doc_type'],
                'total_pages': data['metadata']['total_pages'],
                'indexed_pages': len(data['pages']),
                'total_chunks': len(data['chunks']),
                'classification': data['metadata']['classification'],
            }

            # Add aggregated metadata (only if non-empty)
            if data['jac_regs']:
                doc_info['jac_regulations'] = sorted(data['jac_regs'])
            if data['chapters']:
                doc_info['chapters'] = sorted(data['chapters'])
            if data['safety_keywords']:
                doc_info['safety_keywords'] = sorted(data['safety_keywords'])
            if data['acronyms']:
                doc_info['acronyms'] = sorted(data['acronyms'])[:20]  # Limit for readability
            if data['sms_terms']:
                doc_info['sms_terms'] = sorted(data['sms_terms'])

            documents.append(doc_info)

        return sorted(documents, key=lambda x: x['filename'])

    except Exception as e:
        raise Exception(f"Failed to aggregate documents: {str(e)}")


# ============================================================================
# FASTMCP TOOLS
# ============================================================================

@mcp.tool()
def query_rag_database(query: str, max_results: int = 10) -> Dict[str, Any]:
    """
    Search the technical document database for relevant information.

    This tool searches through indexed technical documents (JAC regulations, safety manuals,
    OSHEMS documents, etc.) using a hybrid approach that combines:
    - Semantic search: Understands the meaning of your query
    - Keyword search: Matches specific terms and acronyms

    When to use this tool:
    - You need to find specific regulations, procedures, or safety information
    - You want to look up technical terms, equipment codes, or JAC acronyms
    - You need to locate requirements related to ammunition, explosives, or safety management
    - You want to find information about OSHEMS, SMS, risk management, or safety investigations

    How it works:
    The tool will return relevant text chunks from the documents along with detailed metadata
    including page numbers, document types, JAC regulation numbers, chapters, safety keywords,
    and more. Each result includes a relevance score (higher is better).

    Tips for better results:
    - Use specific terms: "JAC REG 385-7" is better than "safety regulation"
    - Include context: "ammunition storage procedures" vs just "storage"
    - Try acronyms: "OSHEMS SMS procedures" will find relevant safety management content
    - If you get too few results, try broader terms
    - If results aren't relevant, try more specific terminology

    IMPORTANT: Do NOT query more than 2 times to fetch data. If you need more information
    after 2 queries, ask the user for clarification or to refine their question.

    Args:
        query: Your search question or keywords. Be as specific as possible.
               Examples: "What are the safety distance requirements for ammunition storage?"
                        "JAC REG 385-7 explosive handling procedures"
                        "OSHEMS risk assessment requirements"

        max_results: Maximum number of results to return (1-10). Default is 10.
                    Use fewer results (3-5) for very specific queries.
                    Use more results (8-10) for broader exploratory searches.

    Returns:
        A dictionary containing:
        - results: List of relevant text chunks with metadata and scores
        - query_info: Information about your query including warning if applicable
        - total_found: Total number of results found

        Each result contains:
        - text: The relevant text chunk from the document
        - score: Relevance score (0.0 to 1.0, higher is more relevant)
        - metadata: Document information including:
            - filename: Name of the source PDF
            - pdf_title: Title of the document
            - page_number: Page where this text appears
            - total_pages: Total pages in the source document
            - doc_type: Type of document (e.g., "JAC Regulation", "OSHEMS", "safety")
            - section_title: Section or chapter heading (if available)
            - jac_reg_numbers: JAC regulation numbers found (if any)
            - chapters: Chapter references (if any)
            - safety_keywords: Safety-related keywords (WARNING, CAUTION, DANGER, NOTE)
            - acronyms: Technical acronyms found in this chunk
            - sms_terms: Safety management system terms
            - classification: Document classification (e.g., "FOUO")

    Example usage:
        To find ammunition storage requirements:
        query_rag_database("ammunition storage safety distance requirements", max_results=5)

        To look up a specific regulation:
        query_rag_database("JAC REG 385-7 Chapter 3", max_results=3)
    """
    try:
        # Validate inputs
        if not query or not query.strip():
            return {
                'error': 'Query cannot be empty. Please provide a search question or keywords.',
                'results': [],
                'total_found': 0
            }

        max_results = max(1, min(10, max_results))  # Clamp to 1-10

        # Load RAG components
        model, collection, bm25_data = load_rag_components()

        # Perform hybrid search
        results = perform_hybrid_search(
            query,
            model,
            collection,
            bm25_data,
            top_k=max_results
        )

        # Return results
        return {
            'results': results,
            'total_found': len(results)
        }

    except Exception as e:
        return {
            'error': f'Search failed: {str(e)}',
            'results': [],
            'total_found': 0
        }


@mcp.tool()
def list_documents() -> Dict[str, Any]:
    """
    List all documents currently indexed in the RAG database.

    This tool provides an overview of all technical documents that have been indexed
    and are available for searching. It shows document-level information aggregated
    from all the chunks stored in the database.

    When to use this tool:
    - You want to see what documents are available before searching
    - You need to know the scope of the knowledge base
    - You want to understand what types of documents are indexed
    - You need to verify if a specific document has been indexed
    - You want to see document statistics (pages, chunks, etc.)

    What you'll get:
    For each document, you'll see:
    - Filename and title
    - Document type (JAC Regulation, OSHEMS, safety manual, etc.)
    - Total pages and how many are indexed
    - Number of searchable text chunks
    - Classification (e.g., FOUO)
    - Aggregated metadata: JAC regulations mentioned, chapters, safety keywords,
      acronyms used, and SMS terms (if applicable)

    This information helps you:
    - Understand what's available before crafting specific queries
    - Reference specific documents by name when searching
    - Verify coverage of certain topics or regulation numbers
    - Get a sense of the document structure (chapters, sections, etc.)

    Returns:
        A dictionary containing:
        - documents: List of all indexed documents with detailed metadata
        - summary: High-level statistics about the document collection

        Each document includes:
        - filename: Name of the source PDF file
        - pdf_title: Title of the document (extracted or from filename)
        - doc_type: Type classification (e.g., "JAC Regulation", "OSHEMS")
        - total_pages: Total number of pages in the original PDF
        - indexed_pages: Number of pages that were successfully indexed
        - total_chunks: Number of searchable text chunks created
        - classification: Security classification if applicable (e.g., "FOUO")
        - jac_regulations: List of JAC regulation numbers found (if any)
        - chapters: List of chapters/sections found (if any)
        - safety_keywords: Safety-related terms found (WARNING, CAUTION, etc.)
        - acronyms: Technical acronyms used in the document (up to 20 most common)
        - sms_terms: Safety management system terms found (if any)

    Example usage:
        To see all available documents:
        list_documents()

        Then use the information to craft specific queries like:
        query_rag_database("JAC REG 385-7 from document about ammunition")

    Note: This operation may take a few seconds as it aggregates data from all chunks.
    """
    try:
        # Load ChromaDB collection
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)

        # Aggregate documents
        documents = aggregate_documents(collection)

        # Create summary statistics
        summary = {
            'total_documents': len(documents),
            'total_chunks': sum(doc['total_chunks'] for doc in documents),
            'total_pages': sum(doc['total_pages'] for doc in documents),
            'document_types': list(set(doc['doc_type'] for doc in documents)),
        }

        return {
            'documents': documents,
            'summary': summary
        }

    except Exception as e:
        return {
            'error': f'Failed to list documents: {str(e)}',
            'documents': [],
            'summary': {}
        }


# ============================================================================
# SERVER ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Run the FastMCP server.

    The server will be available for MCP clients (like Claude Desktop) to connect.

    Usage:
        python simple-mcp.py

    Configuration for Claude Desktop (add to config):
    {
      "mcpServers": {
        "rag-documents": {
          "command": "python",
          "args": ["C:\\Users\\arif\\WebstormProjects\\sharepoint-downloader\\simple-mcp.py"]
        }
      }
    }
    """
    print("🚀 Starting RAG Document Search MCP Server...")
    print(f"\n📂 Configuration:")
    print(f"   Models directory: {Path(MODELS_DIR).absolute()}")
    print(f"   Database: {Path(VECTOR_DB_PATH).absolute()}")
    print(f"   BM25 index: {Path(BM25_INDEX_PATH).absolute()}")
    print(f"   Embedding Model: {EMBEDDING_MODEL}")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"\n🔒 Offline Mode: {'ENABLED' if USE_LOCAL_MODELS_ONLY else 'DISABLED'}")
    print(f"   HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE', 'not set')}")
    print(f"   TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE', 'not set')}")
    print(f"   HF_DATASETS_OFFLINE: {os.environ.get('HF_DATASETS_OFFLINE', 'not set')}")
    print("\n💡 Available tools:")
    print("   - query_rag_database: Search documents with hybrid search")
    print("   - list_documents: List all indexed documents")
    print("\n✅ Server ready for MCP connections\n")

    # Run the FastMCP server
    mcp.run()
