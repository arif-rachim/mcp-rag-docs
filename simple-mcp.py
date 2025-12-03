#!/usr/bin/env python3
"""FastMCP Server for RAG Document Search"""
import os, sys, pickle, re
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

# Config
VECTOR_DB_PATH = './chroma_store'
BM25_INDEX_PATH = './bm25_index.pkl'
MODELS_DIR = Path('./models')
EMBEDDING_MODEL_PATH = MODELS_DIR / 'multilingual-e5-large'
RERANKER_MODEL_PATH = MODELS_DIR / 'bge-reranker-v2-m3'
COLLECTION_NAME = 'technical_documents'
SEMANTIC_WEIGHT = 0.7
USE_RERANKER = True  # Set to False to disable reranking

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

mcp = FastMCP(
    "JAC Safety & Regulations Knowledge Base",
    instructions="""This MCP server provides access to UAE Joint Aviation Command (JAC) safety regulations and technical documents (~205 documents in Arabic and English).

KNOWLEDGE BASE CONTAINS:
- JAC REG 385 series (385-1 to 385-10): OSHEMS, aviation safety, ammunition/explosives, risk management, environmental management, emergency management, contractor safety
- JAC Safety Guidelines (SGL): Emergency response, hazmat, helipads, flying displays
- Base-specific Safety Operating Programs (ASC, FUJ, FWG, G10, G18, HAZ, JAI, NAG, SAB, SAR, SAS)
- General safety instructions (aviation, ammunition storage, military displays, weather)
- Safety policies and procedures (Arabic)
- JAC Safety Magazines (2015, 2017, 2019, 2023)
- Environmental awareness materials
- Vehicle and traffic safety guidelines
- UAE Life Safety Code 2018

USE THIS TOOL FOR ANY QUESTIONS ABOUT:
- JAC regulations and compliance requirements
- Aviation safety procedures and programs
- Ammunition, explosives, and weapons safety
- OSHEMS (Occupational Safety, Health & Environmental Management)
- Risk assessment and management
- Safety investigations and incident reporting
- Emergency management and response plans
- Environmental protection and management
- Contractor safety requirements
- Base-specific safety procedures
- Vehicle and traffic safety
- Weather-related safety measures

CITATION REQUIREMENTS (CRITICAL):
You MUST include citations for EVERY fact, requirement, or procedure:
- Format: [JAC REG 385-7, Page 23] or [Safety_Manual.pdf, Page 15]
- Always include document name (JAC REG number or filename) + page number
- Multiple citations if information comes from different sources

Example: "Ammunition storage facilities must maintain minimum safety distances [JAC REG 385-7, Page 23] and comply with environmental requirements [JAC REG 385-6, Page 15]."

ALWAYS use search_documents() for ANY safety or regulation query."""
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
            tokenizer_kwargs={'clean_up_tokenization_spaces': True}
        )

    if _collection is None:
        client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        _collection = client.get_collection(name=COLLECTION_NAME)

    if _bm25_data is None and Path(BM25_INDEX_PATH).exists():
        with open(BM25_INDEX_PATH, 'rb') as f:
            _bm25_data = pickle.load(f)

    if _reranker is None and USE_RERANKER and RERANKER_MODEL_PATH.exists():
        try:
            _reranker = CrossEncoder(
                str(RERANKER_MODEL_PATH),
                max_length=512,
                device='cpu'
            )
        except:
            pass  # Fall back to no reranking

    return _model, _collection, _bm25_data, _reranker

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
        for idx in top_idx:
            if scores[idx] > 0:
                results.append({
                    'text': bm25_data['chunks'][idx],
                    'metadata': {},
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
    if reranker and USE_RERANKER and len(unique) > top_k:
        # Prepare query-document pairs for reranking
        pairs = [[query, r['text'][:512]] for r in unique]  # Limit text length for speed
        rerank_scores = reranker.predict(pairs)

        # Update scores with reranking
        for i, score in enumerate(rerank_scores):
            unique[i]['rerank_score'] = float(score)

        # Sort by rerank score
        unique = sorted(unique, key=lambda x: x.get('rerank_score', x['score']), reverse=True)

    return unique[:top_k]

@mcp.tool(description="""Search UAE JAC safety regulations and technical documents (~205 docs covering aviation safety, ammunition/explosives, OSHEMS, environmental, risk management, emergency response, and base-specific procedures).

SEARCH CAPABILITIES:
- JAC REG 385 series (385-1 OSHEMS, 385-2 Safety Programs, 385-3 Investigations, 385-4 Aviation, 385-5 Risk Mgmt, 385-6 Environmental, 385-7 Ammunition/Explosives, 385-8 Operations, 385-9 Contractors, 385-10 Emergency)
- JAC Safety Guidelines (emergency response, hazmat, helipads, flying displays)
- Base safety programs (ASC, FUJ, FWG, G10, G18, HAZ, JAI, NAG, SAB, SAR, SAS)
- General safety instructions (Arabic & English)
- Environmental awareness, vehicle safety, weather safety
- UAE Life Safety Code 2018

QUERY EXAMPLES:
- "JAC REG 385-7 ammunition storage distance requirements"
- "OSHEMS risk assessment procedures"
- "emergency response plan for aviation incidents"
- "FWG base safety operating procedures"
- "hazardous materials handling requirements"
- "contractor safety management"

HOW TO USE:
- Be specific with JAC REG numbers or topics
- Use technical terms: OSHEMS, HAZMAT, SOP, SMS
- For broad topics use max_results=15-25, specific queries use 5-10
- Searches in both English and Arabic content

METADATA RETURNED:
- filename: Document name (e.g., "JAC REG 385-7 Ammunition...")
- page: Page number where text appears
- jac_reg: JAC REG number if found (e.g., "JAC REG 385-7")
- chapters: Chapter/section references (e.g., "CHAPTER 3")
- safety: Safety keywords (WARNING, CAUTION, DANGER, NOTE)
- lang: Language (en/ar/table)
- score: Relevance score (higher = more relevant)

CITATION REQUIREMENT (MANDATORY):
ALWAYS cite sources: [JAC REG 385-7, Page 23] or [filename.pdf, Page 15]
Include citation for EVERY fact, requirement, or procedure in your response.

Args:
    query: Search query - be specific with JAC REG numbers or technical terms
    max_results: Number of results (1-25, default 10). Use 15-25 for comprehensive coverage.

Returns:
    results: Text chunks with metadata (filename, page, jac_reg, chapters, safety, score)
    total_found: Number of results returned
    query: The search query used""")
def search_documents(query: str, max_results: int = 10) -> Dict[str, Any]:
    try:
        if not query.strip():
            return {'error': 'Query cannot be empty', 'results': [], 'total_found': 0}

        max_results = max(1, min(25, max_results))
        results = hybrid_search(query, top_k=max_results)

        return {
            'results': results,
            'total_found': len(results),
            'query': query
        }
    except Exception as e:
        return {'error': str(e), 'results': [], 'total_found': 0}

@mcp.tool(description="""List all UAE JAC safety regulations and technical documents in the knowledge base (~205 documents).

USE THIS TO:
- See what JAC REG 385 regulations are available (385-1 through 385-10)
- Find available base-specific safety programs (ASC, FUJ, FWG, G10, G18, HAZ, JAI, NAG, SAB, SAR, SAS)
- Check which safety guidelines, environmental materials, or vehicle safety docs are indexed
- Verify if a specific document or regulation has been indexed
- Understand the scope before searching

DOCUMENT TYPES AVAILABLE:
- JAC REG 385 series: OSHEMS, safety programs, investigations, aviation, risk management, environmental, ammunition/explosives, operations, contractors, emergency
- JAC Safety Guidelines (SGL): Emergency response, hazmat, helipads, flying displays
- Base Safety Operating Programs: 11+ military bases
- General safety instructions (Arabic & English)
- Safety policies and procedures
- JAC Safety Magazines (2015, 2017, 2019, 2023)
- Environmental awareness materials
- Vehicle and traffic safety
- UAE Life Safety Code 2018

METADATA RETURNED FOR EACH DOCUMENT:
- filename: PDF filename
- total_pages: Total pages in original document
- indexed_pages: Pages successfully indexed and searchable
- chunks: Number of searchable text segments
- language: Document language (en=English, ar=Arabic, table=Tables)
- jac_regulations: JAC REG numbers found in document (if any)
- chapters: Chapters/sections identified (if any)
- safety_keywords: Safety terms found (WARNING, CAUTION, DANGER, NOTE)

SUMMARY STATISTICS:
- total_documents: Total indexed documents (~205)
- total_chunks: Total searchable text segments
- total_pages: Total pages indexed

Returns:
    documents: List of all documents with detailed metadata (sorted by filename)
    summary: Collection statistics""")
def list_documents() -> Dict[str, Any]:
    try:
        _, collection, _ = load_components()
        data = collection.get()

        if not data or not data['metadatas']:
            return {'documents': [], 'summary': {}}

        # Aggregate by filename
        docs = defaultdict(lambda: {
            'chunks': 0, 'pages': set(), 'metadata': {},
            'jac_regs': set(), 'chapters': set(), 'safety': set()
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
            for key in ['jac_reg', 'chapters', 'safety']:
                if key in meta and meta[key]:
                    doc[key].update(meta[key].split(', '))

        # Format output
        documents = []
        for filename, d in docs.items():
            doc_info = {
                'filename': filename,
                'total_pages': d['metadata']['total_pages'],
                'indexed_pages': len(d['pages']),
                'chunks': d['chunks'],
                'language': d['metadata']['lang']
            }

            if d['jac_regs']:
                doc_info['jac_regulations'] = sorted(d['jac_regs'])
            if d['chapters']:
                doc_info['chapters'] = sorted(d['chapters'])
            if d['safety']:
                doc_info['safety_keywords'] = sorted(d['safety'])

            documents.append(doc_info)

        summary = {
            'total_documents': len(documents),
            'total_chunks': sum(d['chunks'] for d in documents),
            'total_pages': sum(d['total_pages'] for d in documents)
        }

        return {
            'documents': sorted(documents, key=lambda x: x['filename']),
            'summary': summary
        }

    except Exception as e:
        return {'error': str(e), 'documents': [], 'summary': {}}

if __name__ == "__main__":
    print(f"RAG MCP Server")
    print(f"Database: {VECTOR_DB_PATH}")
    print(f"Model: {EMBEDDING_MODEL_PATH.name}")
    print(f"Tools: search_documents, list_documents\n")
    mcp.run()
