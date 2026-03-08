import tiktoken

def get_token_count(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def analyze_chunks(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    chunks = [c.strip() for c in content.split('@@@@@') if c.strip()]
    
    if not chunks:
        print("No chunks found!")
        return

    chunk_data = []
    for i, chunk in enumerate(chunks):
        token_count = get_token_count(chunk)
        title = chunk.split('\n')[0][:50] 
        chunk_data.append({
            "id": i + 1,
            "tokens": token_count,
            "title": title,
            "text": chunk
        })

    biggest = max(chunk_data, key=lambda x: x['tokens'])

    print(f"--- Analysis Results ---")
    print(f"Total Chunks: {len(chunks)}")
    print(f"Biggest Chunk: #{biggest['id']} ({biggest['title']}...)")
    print(f"Token Count:   {biggest['tokens']} tokens")
    print(f"Character Count: {len(biggest['text'])} chars")
    
    if biggest['tokens'] > 2048:
        print("\n⚠️ WARNING: This chunk exceeds the 2k limit!")
    else:
        print("\n✅ This chunk fits within the 2k context window.")


analyze_chunks('md_chunks.txt')