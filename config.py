#!/usr/bin/env python3
"""Shared configuration for RAG document search system"""
from pathlib import Path

# Model storage
MODELS_DIR = Path('./models')
EMBEDDING_MODEL_PATH = MODELS_DIR / 'multilingual-e5-large'
RERANKER_MODEL_PATH = MODELS_DIR / 'bge-reranker-v2-m3'
NER_MODEL_PATH = MODELS_DIR / 'ner' / 'bert-base-multilingual-cased-ner-hrl'

# HuggingFace model IDs (for downloading)
EMBEDDING_MODEL_ID = 'intfloat/multilingual-e5-large'
RERANKER_MODEL_ID = 'BAAI/bge-reranker-v2-m3'
NER_MODEL_ID = 'Davlan/bert-base-multilingual-cased-ner-hrl'

# Database paths
VECTOR_DB_PATH = './chroma_store'
BM25_INDEX_PATH = './bm25_index.pkl'
COLLECTION_NAME = 'technical_documents'

# Search configuration
SEMANTIC_WEIGHT = 0.7
USE_RERANKER = True

# Data folders
DOWNLOADS_FOLDER = './downloads'
