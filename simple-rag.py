#!/usr/bin/env python3
import os, sys, re, pickle
from pathlib import Path
import warnings

# Suppress ALL warnings including tokenizer regex warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer
import fitz
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import logging

# Disable transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

# Config
INPUT_FOLDER = './downloads'
VECTOR_DB_PATH = './chroma_store'
BM25_INDEX_PATH = './bm25_index.pkl'
MODELS_DIR = Path('./models')
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
SEARCH_MODE = 'hybrid'
SEMANTIC_WEIGHT = 0.7

# Local model paths (offline mode)
EMBEDDING_MODEL_PATH = MODELS_DIR / 'multilingual-e5-large'
NER_MODEL_PATH = MODELS_DIR / 'ner' / 'bert-base-multilingual-cased-ner-hrl'
COLLECTION_NAME = 'technical_documents'

# Force offline mode - no internet calls
os.environ.update({
    'HF_DATASETS_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'HF_HUB_OFFLINE': '1',
    'HF_HUB_DISABLE_TELEMETRY': '1'
})

# Bilingual patterns (Arabic + English)
PATTERNS = {
    'jac_reg': r'JAC\s+REG\s+\d+-[A-Z0-9]*',
    'chapters': r'(?:CHAPTER|الفصل|الباب)\s+\d+',
    'sections': r'(?:Section|القسم|المادة)\s+\d+(?:\.\d+)*',
    'safety': r'\b(?:WARNING|CAUTION|DANGER|NOTE|تحذير|تنبيه|خطر|ملاحظة)\b',
    'procedures': r'(?:Procedure|Step|إجراء|خطوة)\s+\d+',
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
            current = current[-CHUNK_OVERLAP:] + para + "\n\n"

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

def build_bm25(chunks):
    """Build BM25 with improved tokenization for Arabic"""
    # Simple but effective tokenization for both languages
    tokenized = [re.findall(r'\w+', chunk.lower()) for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_INDEX_PATH, 'wb') as f:
        pickle.dump({'bm25': bm25, 'chunks': chunks}, f)
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
        return [(data['chunks'][i], scores[i]) for i in top_indices if scores[i] > 0]
    except:
        return []

def hybrid_search(query, model, collection, top_k=5):
    """Optimized hybrid search with better scoring"""
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
        max_score = max(s for _, s in bm25_results) if bm25_results else 1
        for doc, score in bm25_results:
            if max_score > 0:
                results.append({
                    'text': doc,
                    'meta': {},
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

    return unique[:top_k]

def process_pdf(pdf_path, ner, embed_model, client, progress_bar=None):
    """Process PDF and store directly to database"""
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

def search(query, top_k=5):
    """Search with language detection and optimized display"""
    query_lang = detect_language(query)
    print(f"Searching [{query_lang.upper()}]: '{query[:80]}{'...' if len(query) > 80 else ''}'")

    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except:
        print("No index found. Run indexing first.")
        return

    # Load embedding model from local path (offline)
    if not EMBEDDING_MODEL_PATH.exists():
        print(f"Model not found: {EMBEDDING_MODEL_PATH}")
        print("Run: python simple-model-downloader.py")
        return

    model = SentenceTransformer(
        str(EMBEDDING_MODEL_PATH),
        device='cpu',
        local_files_only=True,
        tokenizer_kwargs={'clean_up_tokenization_spaces': True}
    )

    if SEARCH_MODE == 'hybrid':
        results = hybrid_search(query, model, collection, top_k)
    elif SEARCH_MODE == 'keyword':
        results = [{'text': d, 'meta': {}, 'score': s, 'type': 'keyword'} for d, s in bm25_search(query, top_k)]
    else:
        sem = semantic_search(query, model, collection, top_k)
        results = [{'text': d, 'meta': m, 'score': 1 - dist, 'type': 'semantic'}
                   for d, m, dist in zip(sem['documents'][0], sem['metadatas'][0], sem['distances'][0])]

    print(f"\nTop {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        m = r.get('meta', {})
        lang_tag = f"[{m.get('lang', '?').upper()}]" if m.get('lang') else ""

        # Display with total pages info
        page_info = f"p.{m.get('page', '?')}"
        if m.get('total_pages'):
            page_info += f"/{m.get('total_pages')}"

        print(f"[{i}] {r['score']:.3f} {lang_tag} | {m.get('filename', '?')} {page_info}")

        # Show relevant metadata
        if m.get('chapters'):
            print(f"    📖 {m['chapters']}")
        if m.get('safety'):
            print(f"    ⚠️  {m['safety']}")
        if m.get('persons'):
            print(f"    👤 {m['persons']}")

        text_preview = r['text'][:150].replace('\n', ' ')
        print(f"    {text_preview}...\n")

def main():
    """Main indexing function"""
    print("Indexing PDFs (offline mode)...")

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

    # Load NER model
    ner = None
    try:
        if NER_MODEL_PATH.exists():
            ner = pipeline(
                'ner',
                model=AutoModelForTokenClassification.from_pretrained(str(NER_MODEL_PATH), local_files_only=True),
                tokenizer=AutoTokenizer.from_pretrained(str(NER_MODEL_PATH), local_files_only=True, clean_up_tokenization_spaces=True),
                device=-1,
                aggregation_strategy="simple"
            )
            print(f"NER loaded from {NER_MODEL_PATH.name}")
        else:
            print(f"NER model not found: {NER_MODEL_PATH}")
    except Exception as e:
        print(f"NER skip: {e}")

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

    # Process PDFs with progress bar
    print(f"\nProcessing {len(pdf_files)} PDFs...\n")
    total = 0
    processed = 0
    failed = 0

    with tqdm(total=len(pdf_files), desc="Processing PDFs", unit="file") as pbar:
        for pdf in pdf_files:
            try:
                chunks = process_pdf(pdf, ner, embed_model, client, pbar)
                total += chunks
                processed += 1
            except Exception as e:
                failed += 1
                tqdm.write(f"✗ Error with {pdf.name}: {e}")

            pbar.update(1)
            pbar.set_postfix({
                'processed': processed,
                'failed': failed,
                'chunks': total
            })

    # Build BM25 index
    if total > 0:
        print("\nBuilding BM25 index...")
        collection = client.get_collection(name=COLLECTION_NAME)
        docs = collection.get()
        if docs and docs['documents']:
            build_bm25(docs['documents'])

    print(f"\n✓ Done! Processed {processed}/{len(pdf_files)} files, {total} chunks indexed, {failed} failed")

if __name__ == '__main__':
    search(' '.join(sys.argv[2:])) if len(sys.argv) > 1 and sys.argv[1] == 'search' else main()
