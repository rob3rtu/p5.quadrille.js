import ollama
import os
import re

class RagClass:
    dataset_js = []
    dataset_md = []

    # Each element in the VECTOR_DB_XX will be a tuple (chunk, embedding)
    # The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
    VECTOR_DB_JS = []
    VECTOR_DB_MD = []

    EMBEDDING_MODEL = 'nomic-embed-text'
    LANGUAGE_MODEL = 'llama3'

    LLM_INSTRUCTIONS = '''
        You are an expert coding assistant for the 'p5.quadrille.js' library. 
        Your parametric memory regarding this library is unreliable, so you MUST rely ONLY on the Context provided below to answer the user's question. 

        The Context contains a mix of JavaScript source code and Markdown documentation. 
        Follow these guidelines:
        1. Use the Markdown documentation to explain concepts and provide structural examples.
        2. Use the JavaScript source code to verify exact method names, parameter types, and internal logic.
        3. If you provide code examples, format them clearly using Markdown code blocks.
        4. Synthesize the information naturally. Do not just blindly repeat the chunks.
        
        If the Context does not contain the information needed to answer the request, you must state exactly: 'I cannot find that information in the provided documentation.' 
        Do not make up any new information.
        
        Context:
    '''
    
    def ask(self, query):
        retrieved_chunks = self.retrieve(query)

        print('Retrieved knowledge:')
        for chunk, similarity in retrieved_chunks:
            print(f' - (similarity: {similarity:.2f}) {chunk}')

        formatted_chunks = "\n\n---\n\n".join([chunk for chunk, _ in retrieved_chunks])
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


    
    def retrieve(self, query):
        """ Retrieve top K from each DB, filtering out poor matches """
        query_embedding = ollama.embed(model=self.EMBEDDING_MODEL, input=query)['embeddings'][0]

        top_js = self.get_top_results(self.VECTOR_DB_JS, query_embedding)
        top_md = self.get_top_results(self.VECTOR_DB_MD, query_embedding)

        results = top_js + top_md
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def get_top_results(self, db, query_embedding, top_k=3, threshold=0.50):
        similarities = []
        for chunk, chunk_embedding in db:
            similarity = self.cosine_similarity(query_embedding, chunk_embedding)
                
            if similarity >= threshold:
                similarities.append((chunk, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def load_dataset(self):
        self.dataset_js = self.parse_js('../../src/quadrille.js')
        self.dataset_md = self.parse_md('../../content/docs/')
        print(f'Loaded {len(self.dataset_js) + len(self.dataset_md)} total entries; ({len(self.dataset_js)} JS, {len(self.dataset_md)} MD)')

    def add_chunks_to_db(self):
        """ Create ambeddings and add them to vector db alonside their coresponsing chunk """
        for chunk in self.dataset_js:
            embedding = ollama.embed(model=self.EMBEDDING_MODEL, input=chunk)['embeddings'][0]
            self.VECTOR_DB_JS.append((chunk, embedding))
        
        for chunk in self.dataset_md:
            embedding = ollama.embed(model=self.EMBEDDING_MODEL, input=chunk)['embeddings'][0]
            self.VECTOR_DB_MD.append((chunk, embedding))
            
        print(f'Added {len(self.VECTOR_DB_JS)} JS chunks and {len(self.VECTOR_DB_MD)} MD chunks.')

    def cosine_similarity(self, a, b):
        """ Calculate the cosine similarity between two vectors """
        dot_product = sum([x * y for x, y in zip(a, b)])
        norm_a = sum([x ** 2 for x in a]) ** 0.5
        norm_b = sum([x ** 2 for x in b]) ** 0.5
        return dot_product / (norm_a * norm_b)
    
    def parse_md(self, folder_path):
        """ Split all .md files in chunks by headers and including context like function name. """
        chunks = []

        for root, _, files in os.walk(folder_path):
            for filename in files:
                if not filename.endswith(".md"):
                    continue
                
                filepath = os.path.join(root, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                    # 1. Extract Title & Handle Frontmatter (---)
                    page_title = filename.replace('.md', '').capitalize()
                    frontmatter_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)

                    if frontmatter_match:
                        title_search = re.search(r'title:\s*["\']?([^"\'\n]+)', frontmatter_match.group(1))
                        if title_search:
                            page_title = title_search.group(1)
                        content = content[frontmatter_match.end():]

                    # 2. Split by Markdown Headers
                    raw_chunks = re.split(r'(?=\n#{1,4}\s)', '\n' + content)

                    for chunk in raw_chunks:
                        chunk = chunk.strip()
                        # 3. Filter and Contextualize
                        if len(chunk) > 20:
                            contextualized_chunk = f"Context: {page_title} documentation\n{chunk}"
                            chunks.append(contextualized_chunk)

        return chunks



    def parse_js(self, filepath):
        """ Split the JS file in chunks by function and its jsdoc description """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

            chunks = []
            current_chunk = ""
            
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
                            if brace_depth == 1:
                                chunks.append(current_chunk.strip())
                                current_chunk = ""
                        elif c == '}':
                            brace_depth -= 1
                            if brace_depth == 1:
                                chunks.append(current_chunk.strip())
                                current_chunk = ""
                        elif c == ';':
                            if brace_depth == 1:
                                chunks.append(current_chunk.strip())
                                current_chunk = ""
                    
                    elif in_line_comment and c == '\n':
                        in_line_comment = False
                    
                    elif in_block_comment and c == '*' and next_c == '/':
                        current_chunk += next_c
                        i += 1
                        in_block_comment = False
                        
                i += 1
                
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                
            return [chunk for chunk in chunks if chunk]


rag = RagClass()
rag.load_dataset()
# rag.add_chunks_to_db()
# rag.ask("How can I invert filled and empty cells?")