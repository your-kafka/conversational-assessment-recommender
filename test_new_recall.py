import os
import glob
import re
import json
import asyncio
from retriever import HybridRetriever
from agent import AssessmentAgent

def normalize_string(s):
    """Strips all punctuation, spaces, and casing to ensure exact matching."""
    return re.sub(r'[^a-z0-9]', '', s.lower())

def extract_ground_truth_and_chat(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return [], []
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    ground_truth = set()
    table_pattern = re.compile(r'\|\s*\d+\s*\|\s*([^|]+?)\s*\|')
    for match in table_pattern.findall(content):
        item_name = match.strip()
        item_name = item_name.replace('**', '').replace('_', '')
        if item_name and item_name.lower() != 'name':
            ground_truth.add(normalize_string(item_name))

    messages = []
    user_blocks = re.findall(r'> (.*?)(?=\n\n|\Z)', content, re.DOTALL)
    for msg in user_blocks:
        clean_msg = msg.strip().replace('\n> ', ' ')
        if clean_msg:
            messages.append({"role": "user", "content": clean_msg})
        
    return list(ground_truth), messages

async def evaluate_all():
    print("Loading Retriever and Agent...")
    retriever = HybridRetriever()
    agent = AssessmentAgent()
    
    folder_path = "/Users/lucky/Downloads/GenAI_SampleConversations"
    
    # --- RUN ON ALL 10 FILES ---
    files = sorted(glob.glob(os.path.join(folder_path, "*.md")))
    
    total_gt = 0
    total_hits = 0
    results = []

    for filepath in files:
        filename = os.path.basename(filepath)
        gt_items_normalized, messages = extract_ground_truth_and_chat(filepath)
        
        if not gt_items_normalized:
            continue
            
        print(f"\nEvaluating {filename}...")
        
        try:
            # 1. Agent extracts parameters (HyDE, MIL, Exact Names)
            analysis = await agent.analyze_conversation(messages)
            
            # 2. Wide-net Retrieval (Top 50 candidates)
            retrieved_candidates = retriever.retrieve(parameters=analysis, top_k=50)
            
            # 3. LLM Precision Reranking (Curates Top 10)
            final_curated_docs = await agent.llm_rerank(messages, retrieved_candidates)
            
            # 4. Normalize the curated names before checking
            retrieved_names_normalized = [normalize_string(doc.get("name", "")) for doc in final_curated_docs[:10]]
            
            hits = [name for name in gt_items_normalized if name in retrieved_names_normalized]
            recall = len(hits) / len(gt_items_normalized) if gt_items_normalized else 0
            
            total_gt += len(gt_items_normalized)
            total_hits += len(hits)
            
            missed = [name for name in gt_items_normalized if name not in retrieved_names_normalized]
            
            results.append({
                "file": filename,
                "gt_len": len(gt_items_normalized),
                "hits": len(hits),
                "recall": recall,
                "missed": missed
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            # If rate limit hits on one file, don't crash the whole script
            pass

    print("\n" + "="*70)
    print(f"{'File':<10} | {'GT Items':<10} | {'Hits':<10} | {'Recall@10':<10}")
    print("-" * 70)
    for res in results:
        print(f"{res['file']:<10} | {res['gt_len']:<10} | {res['hits']:<10} | {res['recall'] * 100:.1f}%")
        
    print("-" * 70)
    overall_recall = (total_hits / total_gt) * 100 if total_gt else 0
    print(f"FINAL OVERALL AVERAGE RECALL@10: {overall_recall:.1f}%")
    print("="*70)
    
    print("\nDetailed Misses (for tuning the LLM prompt):")
    for res in results:
        if res['missed']:
            print(f"- {res['file']} missed: {res['missed']}")

if __name__ == "__main__":
    asyncio.run(evaluate_all())