#!/usr/bin/env python3
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForTokenClassification

EMBEDDING_MODEL = 'intfloat/multilingual-e5-large'
NER_MODEL = 'Davlan/bert-base-multilingual-cased-ner-hrl'
MODELS_DIR = Path('./models')

if __name__ == '__main__':
    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / 'ner').mkdir(exist_ok=True)

    # Download embedding model
    embedding_path = MODELS_DIR / EMBEDDING_MODEL.split('/')[-1]
    if not embedding_path.exists():
        print(f"Downloading {EMBEDDING_MODEL}...")
        SentenceTransformer(EMBEDDING_MODEL).save(str(embedding_path))

    # Download NER model
    ner_path = MODELS_DIR / 'ner' / NER_MODEL.split('/')[-1]
    if not ner_path.exists():
        print(f"Downloading {NER_MODEL}...")
        AutoTokenizer.from_pretrained(NER_MODEL).save_pretrained(str(ner_path))
        AutoModelForTokenClassification.from_pretrained(NER_MODEL).save_pretrained(str(ner_path))

    print("Done!")
