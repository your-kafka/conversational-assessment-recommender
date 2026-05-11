import json
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, catalog_path="catalog_index.json", faiss_path="faiss_index.bin"):
        with open(catalog_path, 'r') as f:
            self.catalog = json.load(f)
            
        self.name_to_item = {item["name"]: item for item in self.catalog}
            
        self.index = faiss.read_index(faiss_path)
        # Swapping from bge-large (1.34GB) to all-MiniLM-L6-v2 (80MB)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Build "Semantic Prose" for BM25: Prepend Keys and Name to Description
        tokenized_corpus = []
        for item in self.catalog:
            keys_str = " ".join(item.get("keys", []))
            semantic_prose = f"{keys_str} {item.get('name', '')} {item.get('description', '')}"
            tokenized_corpus.append(self._tokenize(semantic_prose))
            
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text):
        return re.findall(r'\w+', text.lower())

    def retrieve(self, parameters, top_k=50): # WIDER NET: Increased to 50
        exact_names = parameters.get("exact_names", [])
        hyde_documents = parameters.get("hyde_documents", [])
        
        candidates = []
        seen_ids = set()
        
        def add_candidate(idx):
            item = self.catalog[idx]
            if item['entity_id'] not in seen_ids:
                candidates.append(item)
                seen_ids.add(item['entity_id'])

        # 1. THE SYMBOLIC PASS
        for name in exact_names:
            if name in self.name_to_item:
                item = self.name_to_item[name]
                if item['entity_id'] not in seen_ids:
                    candidates.append(item)
                    seen_ids.add(item['entity_id'])
                    
        # 2. MIL MAX-POOLING (Dense + BM25 Hybrid)
        if hyde_documents:
            for doc in hyde_documents:
                if not doc.strip():
                    continue
                    
                # A. Dense Vector Search (Good for conceptual matches)
                vec = self.embedder.encode([doc], normalize_embeddings=True)
                distances, indices = self.index.search(vec, 20) 
                for idx in indices[0]:
                    add_candidate(idx)
                    
                # B. BM25 Lexical Search (Crucial for Entity matches like "Excel", "SQL", "DSI")
                tokenized_query = self._tokenize(doc)
                bm25_scores = self.bm25.get_scores(tokenized_query)
                top_bm25_idx = np.argsort(bm25_scores)[::-1][:20]
                for idx in top_bm25_idx:
                    add_candidate(idx)
                        
        return candidates[:top_k]