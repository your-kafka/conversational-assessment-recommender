# SHL Conversational Assessment Recommender

A high-performance AI agent designed to map complex hiring requirements to the SHL Product Catalog. This project moves beyond simple keyword matching by implementing a multi-stage RAG pipeline optimized for 85.7% recall across diverse hiring scenarios.

## 👤 Author Information
- **Name**: Ramkrishna Rathore
- **University**: IIT Kharagpur
- **Email**: [luckyrathore70495@kgpian.iitkgp.ac.in](mailto:luckyrathore70495@kgpian.iitkgp.ac.in)

---

## 🧠 Architectural Highlights

The system utilizes a **Zero-Footprint, Utility-First** approach to recommendation, ensuring that every suggestion is backed by a mechanical necessity within the hiring domain.

- **Hybrid Retrieval Engine**: Orchestrates a dual-stream search using **FAISS** (Semantic Vector Search) and **Rank-BM25** (Lexical Search) to handle both conceptual queries and exact tool matches (e.g., "Excel", "Salesforce").
- **MIL (Multiple Instance Learning) Pooling**: Decomposes complex, multi-intent user prompts into discrete search vectors. This prevents "context dilution" where one part of a prompt might drown out another.
- **HyDE (Hypothetical Document Embeddings)**: Uses an LLM to generate hypothetical product descriptions. This bridges the "vocabulary gap" between user needs (e.g., "angry clients") and formal catalog terminology ("Phone Simulation").
- **Consultative Intent Logic**: A stateful FastAPI implementation that enforces industry-standard consultative behaviors—clarifying vague intents before providing high-confidence recommendations.
- **Precision Reranking**: A Llama-3.3-70B powered reranker validates the top-50 candidates against specific business rules (e.g., Senior vs. Junior battery requirements).

## 📊 Technical Performance
- **Recall@10**: 85.7% (Validated against `test_new_recall.py`)
- **Latency**: Sub-2s response time (Optimized for Render/Groq)
- **Constraint Adherence**: 100% (Strict turn-cap and schema validation)

## 🛠️ Tech Stack
- **Backend**: FastAPI
- **LLM**: Llama 3.3 70B (via Groq)
- **Embeddings**: BAAI/bge-large-en-v1.5
- **Vector Store**: FAISS (FlatIP Index)
- **Search Logic**: Hybrid (Dense + BM25)

## 📁 Repository Structure
- `main.py`: API endpoints and core state management.
- `agent.py`: Intent analysis and precision LLM reranking.
- `retriever.py`: The Hybrid Retrieval engine.
- `build_index.py`: Script to generate Semantic Prose embeddings.
- `test_new_recall.py`: Evaluation suite for accuracy benchmarking.
- `GenAI_SampleConversations/`: Benchmark datasets (C1-C10).

## 📥 Local Setup
1. **Clone & Install**:
   ```bash
   git clone [https://github.com/](https://github.com/)[your-username]/[your-repo]
   pip install -r requirements.txt