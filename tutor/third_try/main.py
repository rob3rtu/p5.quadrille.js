import ollama
import os
import re

class RagClass:
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
    '''

    dataset_js = []
    dataset_md = []

    # Each element in the VECTOR_DB_XX will be a tuple (chunk, embedding). The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
    # CONVERSATION_HISTORY will store objects like {role: "system|user|assistant|tool", content: "..."}
    VECTOR_DB_JS         = []
    VECTOR_DB_MD         = []
    CONVERSATION_HISTORY = []

    EMBEDDING_MODEL = 'nomic-embed-text'
    LANGUAGE_MODEL = 'llama3'
    
    def ask(self, query):
        multi_query = self.compute_multy_query(query)
        retrieved_chunks = self.retrieve(multi_query)

        # debug and analysis
        with open("retrieved_information.txt", "w") as f:
            for chunk, rrf_score in retrieved_chunks:
                f.write(f'\n\n---\n {chunk}\n\n')

        formatted_chunks = "\n\n---\n\n".join([chunk for chunk, _ in retrieved_chunks])

        messages = [{"role": "system", "content": self.LLM_INSTRUCTIONS}]
        messages.extend(self.CONVERSATION_HISTORY)

        prompt = f'''
        Based on the following Context, answer the User Question.
        
        Context:
        {formatted_chunks}
        
        User Question:
        {query}
        '''

        messages.append({'role': 'user', 'content': prompt})

        stream = ollama.chat(
            model=self.LANGUAGE_MODEL,
            messages=messages,
            stream=True
        )

        print("thinking...")
        response = ""
        for chunk in stream:
            content = chunk['message']['content']
            response += content
            print(content, end='', flush=True)

        self.CONVERSATION_HISTORY.append({'role': 'user', 'content': query})
        self.CONVERSATION_HISTORY.append({'role': 'assistant', 'content': response})

        # debug and analysis
        with open("conversation_history.txt", "w") as f:
            for m in self.CONVERSATION_HISTORY:
                f.write(f'\n\n---\n role: {m["role"]}\n content: {m["content"]}')

    def compute_multy_query(self, query, k=3):
        """ Given a query, reformulate that in k other similar queries. This takes into account conversation history """
        queries = [query]
       
       # using last 3 pairs of (question, response)
        recent_history = self.CONVERSATION_HISTORY[-6:] 
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])

        prompt = f"""
        You are an expert at query expansion for a RAG assistant of the p5.quadrille.js library.
        
        USER CONVERSATION HISTORY:
        {history_text}
        
        CURRENT USER QUERY:
        "{query}"
        
        TASK:
        Generate {k} different versions of the "CURRENT USER QUERY" to improve document retrieval.
        - Use the history in order to replace pronouns, for example "how do I use it?", where "it" could reffer to a previous mention method
        - Use different keywords and sentence structures for each version.
        - Don't use " character to wrap the new queries
        - Output ONLY the {k} queries, one per line. Don't include any message from you, like "Here are {k} queries"!
        """

        messages = []
        messages.extend(self.CONVERSATION_HISTORY)
        messages.append({'role': 'user', 'content': prompt})

        response = ollama.chat(
            model=self.LANGUAGE_MODEL,
            messages=messages,
            stream=False
        )["message"]["content"]

        queries.extend([line.strip() for line in response.split('\n') if line.strip()])

        # debug and analysis
        with open("multi_query.txt", "w") as f:
            for i, q in enumerate(queries):
                f.write(f'\n\n--- {"ORIGINAL QUERY" if i == 0 else ""}\n {q}\n\n')

        return queries

    
    def retrieve(self, multi_query, k=5):
        """ Retrieve top K from each DB, filtering best matches using RRF(Reciprocal Rank Fusion) """
        rrf_scores = {}
        multi_query_embedding = [ollama.embed(model=self.EMBEDDING_MODEL, input=query)['embeddings'][0] for query in multi_query]

        for query_embedding in multi_query_embedding:
            top_js = self.get_top_results(self.VECTOR_DB_JS, query_embedding, top_k=k)
            top_md = self.get_top_results(self.VECTOR_DB_MD, query_embedding, top_k=k)
            
            combined_results = top_js + top_md
            combined_results.sort(key=lambda x: x[1], reverse=True)

            for rank, (chunk, _query_similarity) in enumerate(combined_results, start=1):
                score = 1.0 / (60 + rank)
                
                if chunk in rrf_scores:
                    rrf_scores[chunk] += score
                else:
                    rrf_scores[chunk] = score

        final_results = list(rrf_scores.items())
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:k]
    
    def get_top_results(self, db, query_embedding, top_k=3, threshold=0.50):
        similarities = []
        for chunk, chunk_embedding in db:
            similarity = self.cosine_similarity(query_embedding, chunk_embedding)
                
            if similarity >= threshold:
                similarities.append((chunk, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def load_dataset(self):
        self.dataset_js = self.parse_js('../../src/quadrille.js') + self.parse_js("../../src/addon.js")
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
rag.add_chunks_to_db()

print("✅ The model is ready, please ask a question about p5.quadrille.js")
while True:
    q = input(f"\n\n❔ Ask {rag.LANGUAGE_MODEL}: ")
    if q == "q":
        break

    rag.ask(q)