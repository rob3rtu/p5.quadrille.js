import sys
from indexer import Indexer
from vector_engine import VectorEngine
from generator import LocalGenerator

def main():
    REPO_PATH = "../../"
    MODEL_PATH = "./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf" 

    print("--- Phase 1: Ingestion ---")
    indexer = Indexer(REPO_PATH)
    chunks = indexer.load_and_chunk()
    if not chunks:
        print("Error: No chunks found. Check repo path.")
        sys.exit(1)

    print("\n--- Phase 2: Indexing ---")
    engine = VectorEngine()
    engine.build_index(chunks)

    print("\n--- Phase 3: Model Loading ---")
    generator = LocalGenerator(MODEL_PATH)

    print("\n--- Phase 4: Ready ---")
    print("Type 'exit' to quit.")
    while True:
        query = input("\n>> Ask about p5.quadrille.js: ")
        if query.lower() == 'exit': break
        
        # Retrieval
        context = engine.search(query, k=4)
        
        print(f"\n")
        print(f"Top Match Distance: {context['distance']:.4f}")
        
        answer = generator.generate(query, context)
        print(f"\nResponse:\n{answer}")

if __name__ == "__main__":
    main()