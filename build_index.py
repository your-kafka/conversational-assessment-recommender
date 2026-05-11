import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def build_semantic_prose(item):
    """
    Transforms raw JSON into a structured, semantic narrative string.
    This creates a vastly superior dense vector for FAISS.
    """
    name = item.get("name", "")
    desc = item.get("description", "")
    
    # 1. Categorical Weighting: Prepend the high-level taxonomy (Keys)
    keys = item.get("keys", [])
    keys_str = ", ".join(keys) if keys else "General Assessment"
    
    # 2. Extract meaningful metadata to enrich the text
    duration = item.get("duration", "Time not specified")
    job_levels = ", ".join(item.get("job_levels", [])) if item.get("job_levels") else "All Levels"
    
    # 3. Format as a narrative text block
    prose = f"Category: {keys_str}. Assessment Name: {name}. Target Job Levels: {job_levels}. Duration: {duration}. Description: {desc}."
    
    return prose

def main():
    print("Loading raw catalog...")
    # Replace with the name of your raw, original catalog file
    with open("catalog.json", "r", encoding="utf-8") as f:
        catalog = json.load(f)

    print("Loading BAAI embedding model...")
    # We use the exact same model that retriever.py uses
    embedder = SentenceTransformer('BAAI/bge-large-en-v1.5')

    # 1. Transform the catalog into Semantic Prose strings
    print("Transforming catalog into Semantic Prose...")
    texts_to_embed = [build_semantic_prose(item) for item in catalog]

    # 2. Generate Dense Embeddings
    print(f"Generating embeddings for {len(texts_to_embed)} items. This may take a minute...")
    # normalize_embeddings=True is highly recommended for Cosine Similarity with BAAI models
    embeddings = embedder.encode(texts_to_embed, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # 3. Build the FAISS Index
    print("Building FAISS Index...")
    dimension = embeddings.shape[1]
    # Using IndexFlatIP (Inner Product) since embeddings are normalized, which perfectly mimics Cosine Similarity
    index = faiss.IndexFlatIP(dimension) 
    index.add(embeddings)

    # 4. Save the artifacts
    faiss.write_index(index, "faiss_index.bin")
    
    # Save the cleaned catalog for the retriever to use
    with open("catalog_index.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=4)

    print("✅ Successfully built faiss_index.bin and catalog_index.json!")

if __name__ == "__main__":
    main()