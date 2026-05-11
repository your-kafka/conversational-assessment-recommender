from retriever import HybridRetriever

try:
    print("🚀 Initializing HybridRetriever (MiniLM version)...")
    retriever = HybridRetriever()
    
    # Your 'retrieve' method expects a dictionary with 'hyde_documents'
    test_params = {
        "hyde_documents": ["someone who is good at managing stress and angry customers"],
        "exact_names": []
    }
    
    print(f"\n🔍 Testing Query: {test_params['hyde_documents'][0]}")
    
    # Calling the correct method name 'retrieve'
    results = retriever.retrieve(test_params, top_k=3)
    
    print("\n--- Top Recommendations ---")
    if not results:
        print("No results found. Did you run build_index.py yet?")
    else:
        for i, item in enumerate(results):
            print(f"{i+1}. {item['name']}")
            # Use .get() in case 'description' isn't in your exact catalog structure
            print(f"   Description: {item.get('description', 'No description available')[:75]}...")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    if "dimension" in str(e).lower() or "assert" in str(e).lower():
        print("\n💡 FIX: This is a dimension mismatch! You MUST run 'python build_index.py' first.")