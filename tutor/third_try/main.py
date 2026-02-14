import ollama

class RagClass:
    dataset = []

    # Each element in the VECTOR_DB will be a tuple (chunk, embedding)
    # The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
    VECTOR_DB = []
    EMBEDDING_MODEL = 'nomic-embed-text'
    LANGUAGE_MODEL = 'llama3'

    LLM_INSTRUCTIONS = '''
            You are an expert coding assistant for the 'p5.quadrille.js' library. 
            Your parametric memory regarding this library is unreliable. 
            You MUST rely ONLY on the Context provided below to answer the user's question. 
            If the Context does not contain the specific function or method requested, 
            you must state: 'I cannot find that information in the provided documentation.'
            Don't make up any new information: "
        '''
    
    def ask(self, query):
        retrieved_chunks = self.retrieve(query)

        print('Retrieved knowledge:')
        for chunk, similarity in retrieved_chunks:
            print(f' - (similarity: {similarity:.2f}) {chunk}')

        formatted_chunks = "\n".join([f' - {chunk}' for chunk, _ in retrieved_chunks])
        prompt = f'''
            {self.LLM_INSTRUCTIONS}
            {formatted_chunks}
        '''

        stream = ollama.chat(
            model=self.LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': query},
            ],
            stream=True
        )

        print("LLM response: ")
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)


    
    def retrieve(self, query, top_k=3):
        query_embedding = ollama.embed(model=self.EMBEDDING_MODEL, input=query)['embeddings'][0]

        similarities = []
        for chunk, chunk_embedding in self.VECTOR_DB:
            similarity = self.cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((chunk, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def load_dataset(self):
        self.dataset = self.parse_js_by_function('../../src/quadrille.js')
        print(f'Loaded {len(rag.dataset)} entries')

    def add_chunks_to_db(self):
        """ Create ambeddings and them to vector db alonside their coresponsing chunk """
        for chunk in self.dataset:
            embedding = ollama.embed(model=self.EMBEDDING_MODEL, input=chunk)['embeddings'][0]
            self.VECTOR_DB.append((chunk, embedding))
        print(f'Added {len(rag.VECTOR_DB)} entries in vector db')

    def cosine_similarity(self, a, b):
        """ Calculate the cosine similarity between two vectors """
        dot_product = sum([x * y for x, y in zip(a, b)])
        norm_a = sum([x ** 2 for x in a]) ** 0.5
        norm_b = sum([x ** 2 for x in b]) ** 0.5
        return dot_product / (norm_a * norm_b)


    def parse_js_by_function(self, filepath):
        """ Split the JS file in chunks by function and its jsdoc description """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

            chunks = []
            current_chunk = ""
            
            # State tracking
            brace_depth = 0
            in_string = False
            string_char = ''
            in_line_comment = False
            in_block_comment = False
            
            i = 0
            length = len(content)
            
            while i < length:
                c = content[i]
                next_c = content[i+1] if i + 1 < length else ''
                
                current_chunk += c
                
                # 1. Handle Strings (ignore braces and comments inside quotes)
                if not in_line_comment and not in_block_comment:
                    if c in ('"', "'", "`"):
                        if not in_string:
                            in_string = True
                            string_char = c
                        elif string_char == c:
                            # Check if the quote is escaped (e.g., \")
                            backslashes = 0
                            idx = i - 1
                            while idx >= 0 and content[idx] == '\\':
                                backslashes += 1
                                idx -= 1
                            if backslashes % 2 == 0:
                                in_string = False
                
                # 2. Handle Comments and Braces
                if not in_string:
                    if not in_block_comment and not in_line_comment:
                        if c == '/' and next_c == '/':
                            in_line_comment = True
                        elif c == '/' and next_c == '*':
                            in_block_comment = True
                        elif c == '{':
                            brace_depth += 1
                            # Split right after the class declaration opens
                            if brace_depth == 1:
                                chunks.append(current_chunk.strip())
                                current_chunk = ""
                        elif c == '}':
                            brace_depth -= 1
                            # Split when a method ends (depth drops back to class level)
                            if brace_depth == 1:
                                chunks.append(current_chunk.strip())
                                current_chunk = ""
                        elif c == ';':
                            # Split when a property declaration (like static _textColor) ends
                            if brace_depth == 1:
                                chunks.append(current_chunk.strip())
                                current_chunk = ""
                    
                    # Reset line comment at the end of the line
                    elif in_line_comment and c == '\n':
                        in_line_comment = False
                    
                    # End block comment
                    elif in_block_comment and c == '*' and next_c == '/':
                        current_chunk += next_c
                        i += 1
                        in_block_comment = False
                        
                i += 1
                
            # Append whatever is left at the end of the file (like exports)
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                
            # Filter out any accidentally empty chunks
            return [chunk for chunk in chunks if chunk]


rag = RagClass()
rag.load_dataset()
rag.add_chunks_to_db()
rag.ask("How can I invert filled and empty cells?")