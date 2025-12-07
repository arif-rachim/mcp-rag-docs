#!/usr/bin/env python3
import os, sys, re, pickle
from pathlib import Path
import warnings

# Suppress ALL warnings including tokenizer regex warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import fitz
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

# Disable transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

# Import shared configuration
from config import (
    VECTOR_DB_PATH,
    BM25_INDEX_PATH,
    MODELS_DIR,
    EMBEDDING_MODEL_PATH,
    NER_MODEL_PATH,
    RERANKER_MODEL_PATH,
    COLLECTION_NAME,
    SEMANTIC_WEIGHT,
    USE_RERANKER,
    DOWNLOADS_FOLDER as INPUT_FOLDER
)

# Processing-specific constants (keep local)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
NUM_WORKERS = min(cpu_count() - 1, 8) or 1  # Use CPU-1 cores, max 8 workers

# Force offline mode - no internet calls
os.environ.update({
    'HF_DATASETS_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_OFFLINE': '1',
    'HF_HUB_DISABLE_TELEMETRY': '1'
})

# Essential patterns for metadata extraction
PATTERNS = {
    'jac_reg': r'JAC\s+REG\s+\d+-\d+',
    'jac_sgl': r'JAC\s+SGL\s+\d+-\d+\.\d+',
    'sop': r'\bSOP\b',
    'procedure': r'\b(?:Procedure|إجراء)\b',
}

def detect_language(text):
    """Detect if text is primarily Arabic or English"""
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text[:500]))
    return 'ar' if arabic_chars > 50 else 'en'

def extract_pdf(pdf_path):
    """Extract text and tables from PDF using PyMuPDF"""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text()
        if text.strip():
            pages.append({'page': page_num, 'text': text, 'lang': detect_language(text)})
        # Extract tables
        for table in page.find_tables():
            extracted = table.extract()
            if extracted:
                table_text = '\n'.join([' | '.join(str(cell) for cell in row if cell) for row in extracted if any(row)])
                if table_text.strip():
                    pages.append({'page': page_num, 'text': f"[TABLE]\n{table_text}", 'lang': 'table'})
    doc.close()
    return pages

def truncate_to_word_boundary(text, max_length, from_end=False):
    """Truncate text to word boundary, supporting English and Arabic"""
    if len(text) <= max_length:
        return text

    if from_end:
        # Take from end: find first space after the cutoff point
        truncated = text[-max_length:]
        space_idx = truncated.find(' ')
        if space_idx > 0:
            return truncated[space_idx + 1:]
        return truncated
    else:
        # Take from start: find last space before the cutoff point
        truncated = text[:max_length]
        space_idx = truncated.rfind(' ')
        if space_idx > 0:
            return truncated[:space_idx]
        return truncated

def chunk_text(text):
    """Smart chunking for Arabic and English"""
    if not text or len(text) < 50:
        return []

    chunks, current = [], ""
    # Split by double newline or Arabic sentence markers
    paragraphs = re.split(r'\n\n+|(?<=[.!?؟।])\s+', text)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) < CHUNK_SIZE:
            current += para + "\n\n"
        else:
            if current.strip():
                chunks.append(current.strip())
            overlap_text = truncate_to_word_boundary(current, CHUNK_OVERLAP, from_end=True)
            current = overlap_text + para + "\n\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text[:CHUNK_SIZE]]

def extract_patterns(text):
    """Extract bilingual metadata patterns"""
    return {k: ', '.join(list(set(re.findall(p, text, re.IGNORECASE | re.UNICODE)))[:5])
            for k, p in PATTERNS.items() if re.findall(p, text, re.IGNORECASE | re.UNICODE)}

def create_embeddings(texts, model):
    """Create embeddings with batching for efficiency"""
    prefixed = [f"passage: {t[:512]}" for t in texts]  # Limit to 512 chars for speed
    return model.encode(prefixed, show_progress_bar=False, batch_size=64, normalize_embeddings=True)

def store_chromadb(chunks, embeddings, metadata_list, client):
    """Store with efficient batching"""
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Arabic-English bilingual technical documents"}
    )
    ids = [f"{m.get('filename', 'doc')}_{m.get('page', 0)}_{i}" for i, m in enumerate(metadata_list)]

    # Batch insert for efficiency
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        end = min(i + batch_size, len(chunks))
        collection.add(
            documents=chunks[i:end],
            embeddings=embeddings[i:end].tolist() if hasattr(embeddings, 'tolist') else embeddings[i:end],
            metadatas=metadata_list[i:end],
            ids=ids[i:end]
        )
    return collection

def build_bm25(chunks, metadatas):
    """Build BM25 with improved tokenization for Arabic"""
    # Simple but effective tokenization for both languages
    tokenized = [re.findall(r'\w+', chunk.lower()) for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_INDEX_PATH, 'wb') as f:
        pickle.dump({'bm25': bm25, 'chunks': chunks, 'metadatas': metadatas}, f)
    return bm25

def semantic_search(query, model, collection, top_k=10):
    """Semantic search with normalized embeddings"""
    embedding = model.encode([f"query: {query}"], normalize_embeddings=True)[0]
    return collection.query(query_embeddings=[embedding.tolist()], n_results=top_k)

def bm25_search(query, top_k=10):
    """BM25 keyword search with improved tokenization"""
    try:
        with open(BM25_INDEX_PATH, 'rb') as f:
            data = pickle.load(f)
        # Tokenize query same way as chunks
        query_tokens = re.findall(r'\w+', query.lower())
        scores = data['bm25'].get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:top_k]

        # Return chunks with metadata (handle legacy indices without metadatas)
        metadatas = data.get('metadatas', [])
        if metadatas:
            return [(data['chunks'][i], metadatas[i], scores[i]) for i in top_indices if scores[i] > 0]
        else:
            # Fallback for old indices without metadata
            return [(data['chunks'][i], {}, scores[i]) for i in top_indices if scores[i] > 0]
    except:
        return []

def hybrid_search(query, model, collection, top_k=5, reranker=None):
    """Optimized hybrid search with optional reranking"""
    results = []

    # Semantic search
    sem = semantic_search(query, model, collection, top_k * 2)
    for doc, meta, dist in zip(sem['documents'][0], sem['metadatas'][0], sem['distances'][0]):
        results.append({
            'text': doc,
            'meta': meta,
            'score': (1 - dist) * SEMANTIC_WEIGHT,
            'type': 'semantic'
        })

    # BM25 keyword search
    bm25_results = bm25_search(query, top_k * 2)
    if bm25_results:
        max_score = max(s for _, _, s in bm25_results) if bm25_results else 1
        for doc, meta, score in bm25_results:
            if max_score > 0:
                results.append({
                    'text': doc,
                    'meta': meta,
                    'score': (score / max_score) * (1 - SEMANTIC_WEIGHT),
                    'type': 'keyword'
                })

    # Deduplicate and rank
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

def extract_and_chunk_pdf(pdf_path, use_ner=False):
    """Worker function: Extract and chunk PDF (no embedding/storage)
    Returns: (filename, chunks_list, metadata_list) or None on error
    """
    try:
        filename = os.path.basename(pdf_path)
        pages = extract_pdf(pdf_path)

        if not pages:
            return None

        # Get total page count
        total_pages = max(p['page'] for p in pages)

        all_chunks, all_meta = [], []

        # Load NER model only if requested (in worker process)
        ner = None
        if use_ner and NER_MODEL_PATH.exists():
            try:
                ner = pipeline(
                    'ner',
                    model=AutoModelForTokenClassification.from_pretrained(str(NER_MODEL_PATH), local_files_only=True),
                    tokenizer=AutoTokenizer.from_pretrained(str(NER_MODEL_PATH), local_files_only=True, clean_up_tokenization_spaces=True),
                    device=-1,
                    aggregation_strategy="simple"
                )
            except:
                pass

        for page_data in pages:
            chunks = chunk_text(page_data['text'])

            for idx, chunk in enumerate(chunks):
                meta = extract_patterns(chunk)
                meta.update({
                    'filename': filename,
                    'page': page_data['page'],
                    'total_pages': total_pages,
                    'chunk': idx,
                    'lang': page_data.get('lang', 'unknown')
                })

                # NER on first chunk of each page
                if ner and idx == 0 and len(chunk) > 100:
                    try:
                        entities = ner(chunk[:500], aggregation_strategy="simple")
                        persons = [e['word'] for e in entities if 'PER' in e['entity_group']]
                        orgs = [e['word'] for e in entities if 'ORG' in e['entity_group']]
                        if persons:
                            meta['persons'] = ', '.join(set(persons)[:3])
                        if orgs:
                            meta['orgs'] = ', '.join(set(orgs)[:3])
                    except:
                        pass

                all_chunks.append(chunk)
                all_meta.append(meta)

        return (filename, all_chunks, all_meta)

    except Exception as e:
        return None

def process_pdf(pdf_path, ner, embed_model, client, progress_bar=None):
    """Legacy function: Process PDF and store directly to database (sequential mode)"""
    filename = os.path.basename(pdf_path)
    if progress_bar:
        progress_bar.set_description(f"Processing {filename[:50]}")

    pages = extract_pdf(pdf_path)
    if not pages:
        return 0

    # Get total page count (excluding tables)
    total_pages = max(p['page'] for p in pages)

    all_chunks, all_meta = [], []

    for page_data in pages:
        chunks = chunk_text(page_data['text'])

        for idx, chunk in enumerate(chunks):
            meta = extract_patterns(chunk)
            meta.update({
                'filename': filename,
                'page': page_data['page'],
                'total_pages': total_pages,
                'chunk': idx,
                'lang': page_data.get('lang', 'unknown')
            })

            # NER on first chunk of each page
            if ner and idx == 0 and len(chunk) > 100:
                try:
                    entities = ner(chunk[:500], aggregation_strategy="simple")
                    persons = [e['word'] for e in entities if 'PER' in e['entity_group']]
                    orgs = [e['word'] for e in entities if 'ORG' in e['entity_group']]
                    if persons:
                        meta['persons'] = ', '.join(set(persons)[:3])
                    if orgs:
                        meta['orgs'] = ', '.join(set(orgs)[:3])
                except:
                    pass

            all_chunks.append(chunk)
            all_meta.append(meta)

    embeddings = create_embeddings(all_chunks, embed_model)
    store_chromadb(all_chunks, embeddings, all_meta, client)
    return len(all_chunks)

def main(use_multiprocessing=True):
    """Main indexing function with optional multiprocessing"""
    print(f"Indexing PDFs (offline mode, {'PARALLEL' if use_multiprocessing else 'SEQUENTIAL'})...")

    pdf_files = list(Path(INPUT_FOLDER).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs in {INPUT_FOLDER}")
        return

    print(f"Found {len(pdf_files)} PDFs")

    # Delete old ChromaDB and BM25 index to start fresh
    import shutil
    if Path(VECTOR_DB_PATH).exists():
        print(f"Deleting old database: {VECTOR_DB_PATH}")
        shutil.rmtree(VECTOR_DB_PATH)
    if Path(BM25_INDEX_PATH).exists():
        print(f"Deleting old BM25 index: {BM25_INDEX_PATH}")
        Path(BM25_INDEX_PATH).unlink()
    print("Starting fresh indexing...\n")

    # Check if NER is available
    use_ner = NER_MODEL_PATH.exists()
    if use_ner:
        print(f"NER model found: {NER_MODEL_PATH.name}")
    else:
        print(f"NER model not found, skipping entity extraction")

    # Load embedding model
    if not EMBEDDING_MODEL_PATH.exists():
        print(f"Embedding model not found: {EMBEDDING_MODEL_PATH}")
        print("Run: python simple-model-downloader.py")
        return

    print(f"Loading embedding model from {EMBEDDING_MODEL_PATH.name}...")
    embed_model = SentenceTransformer(
        str(EMBEDDING_MODEL_PATH),
        device='cpu',
        local_files_only=True,
        tokenizer_kwargs={'clean_up_tokenization_spaces': True}
    )
    print("Embedding model loaded")

    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    # MULTIPROCESSING MODE
    if use_multiprocessing and len(pdf_files) > 1:
        print(f"\nUsing {NUM_WORKERS} worker processes for parallel PDF extraction...\n")

        # Create worker function with partial to pass use_ner
        worker_func = partial(extract_and_chunk_pdf, use_ner=use_ner)

        # Process PDFs in parallel
        total_chunks = 0
        processed = 0
        failed = 0

        with Pool(processes=NUM_WORKERS) as pool:
            # Use imap_unordered for better performance with progress bar
            with tqdm(total=len(pdf_files), desc="Extracting PDFs", unit="file") as pbar:
                for result in pool.imap_unordered(worker_func, pdf_files):
                    if result is None:
                        failed += 1
                        pbar.update(1)
                        continue

                    filename, chunks, metadata = result
                    processed += 1

                    # Generate embeddings and store in main process
                    if chunks:
                        try:
                            embeddings = create_embeddings(chunks, embed_model)
                            store_chromadb(chunks, embeddings, metadata, client)
                            total_chunks += len(chunks)
                        except Exception as e:
                            tqdm.write(f"✗ Error storing {filename}: {e}")
                            failed += 1

                    pbar.update(1)
                    pbar.set_postfix({
                        'processed': processed,
                        'failed': failed,
                        'chunks': total_chunks
                    })

    # SEQUENTIAL MODE (fallback)
    else:
        print(f"\nProcessing {len(pdf_files)} PDFs sequentially...\n")
        total_chunks = 0
        processed = 0
        failed = 0

        # Load NER in sequential mode
        ner = None
        if use_ner:
            try:
                ner = pipeline(
                    'ner',
                    model=AutoModelForTokenClassification.from_pretrained(str(NER_MODEL_PATH), local_files_only=True),
                    tokenizer=AutoTokenizer.from_pretrained(str(NER_MODEL_PATH), local_files_only=True, clean_up_tokenization_spaces=True),
                    device=-1,
                    aggregation_strategy="simple"
                )
            except Exception as e:
                print(f"NER skip: {e}")

        with tqdm(total=len(pdf_files), desc="Processing PDFs", unit="file") as pbar:
            for pdf in pdf_files:
                try:
                    chunks = process_pdf(pdf, ner, embed_model, client, pbar)
                    total_chunks += chunks
                    processed += 1
                except Exception as e:
                    failed += 1
                    tqdm.write(f"✗ Error with {pdf.name}: {e}")

                pbar.update(1)
                pbar.set_postfix({
                    'processed': processed,
                    'failed': failed,
                    'chunks': total_chunks
                })

    # Build BM25 index
    if total_chunks > 0:
        print("\nBuilding BM25 index...")
        collection = client.get_collection(name=COLLECTION_NAME)
        docs = collection.get()
        if docs and docs['documents']:
            build_bm25(docs['documents'], docs['metadatas'])

    print(f"\n✓ Done! Processed {processed}/{len(pdf_files)} files, {total_chunks} chunks indexed, {failed} failed")

if __name__ == '__main__':
    # Required for multiprocessing on Windows
    import multiprocessing
    multiprocessing.freeze_support()

    if len(sys.argv) > 1 and sys.argv[1] == '--sequential':
        main(use_multiprocessing=False)
    else:
        main(use_multiprocessing=True)
