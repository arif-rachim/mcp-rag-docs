#!/usr/bin/env python3
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForTokenClassification

EMBEDDING_MODEL = 'intfloat/multilingual-e5-large'
NER_MODEL = 'Davlan/bert-base-multilingual-cased-ner-hrl'
RERANKER_MODEL = 'BAAI/bge-reranker-v2-m3'
MODELS_DIR = Path('./models')

if __name__ == '__main__':
    print("Setting up offline models for RAG system...\n")

    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / 'ner').mkdir(exist_ok=True)

    # 1. Download embedding model
    print("[1/3] Embedding Model")
    embedding_path = MODELS_DIR / EMBEDDING_MODEL.split('/')[-1]
    if embedding_path.exists():
        print(f"  ✓ Already exists: {embedding_path.name}")
    else:
        print(f"  Downloading {EMBEDDING_MODEL}...")
        SentenceTransformer(EMBEDDING_MODEL).save(str(embedding_path))
        print(f"  ✓ Saved to: {embedding_path}")

    # 2. Download NER model
    print("\n[2/3] NER Model")
    ner_path = MODELS_DIR / 'ner' / NER_MODEL.split('/')[-1]
    if ner_path.exists():
        print(f"  ✓ Already exists: {ner_path.name}")
    else:
        print(f"  Downloading {NER_MODEL}...")
        AutoTokenizer.from_pretrained(NER_MODEL).save_pretrained(str(ner_path))
        AutoModelForTokenClassification.from_pretrained(NER_MODEL).save_pretrained(str(ner_path))
        print(f"  ✓ Saved to: {ner_path}")

    # 3. Download reranker model
    print("\n[3/3] Reranker Model")
    reranker_path = MODELS_DIR / RERANKER_MODEL.split('/')[-1]
    if reranker_path.exists():
        print(f"  ✓ Already exists: {reranker_path.name}")
    else:
        print(f"  Downloading {RERANKER_MODEL}...")
        print("  (This is ~560MB, may take a few minutes...)")
        reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
        reranker.save(str(reranker_path))
        print(f"  ✓ Saved to: {reranker_path}")

    print("\n" + "="*60)
    print("✓ All models downloaded successfully!")
    print("="*60)
    print(f"\nModels saved to: {MODELS_DIR.absolute()}")
    print(f"  - Embedding: {embedding_path.name} (~2GB)")
    print(f"  - NER: {ner_path.name} (~500MB)")
    print(f"  - Reranker: {reranker_path.name} (~560MB)")
    print(f"\nTotal size: ~3GB")
    print("\nYou can now run the RAG system in offline mode!")
