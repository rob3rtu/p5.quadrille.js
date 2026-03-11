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
        5. When providing code examples, NEVER truncate them with comments like "// Your code goes here". Always provide the complete, working code block exactly as it appears in the Context.
        
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

    EMBEDDING_MODEL = 'embeddinggemma'
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

        print("thinking...\n")
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

    def compute_multy_query(self, query, k=4):
        """ Given a query, reformulate that in k other similar queries. This takes into account conversation history """
        queries = [query]
       
       # using last 3 pairs of (question, response)
        recent_history = self.CONVERSATION_HISTORY[-6:] 
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
        
        if not history_text.strip():
            history_text = "No prior conversation history."

        prompt = f"""
        You are an expert at query expansion for a RAG assistant of the 'p5.quadrille.js' library.
        
        TASK:
        Generate {k} different queries based on the "CURRENT USER QUERY" to improve document retrieval in a vector database. 
        The first {k-1} queries should be highly specific, and the final query MUST be a "step-back" conceptual query.
        
        RULES:
        1. If the user uses "it", "this", or "that", replace it with the specific method or concept from the CONVERSATION HISTORY. If there is no history, ALWAYS assume the user refers to the "p5.quadrille.js" library.
        2. For the specific queries, use different technical keywords for each version (e.g., "install" -> "setup", "npm", "import").
        3. The very last query MUST be a broader, higher-level conceptual question that addresses the fundamental principles, overarching mechanics, or architecture behind the user's specific query.
        4. Output ONLY the {k} queries, one per line. No quotes, no numbering, no text of any kind added by you, like "Here are the queries", just the text of the query.

        EXAMPLES:
        User Query: "why does read(5, 5) return undefined?"
        History: No prior conversation history.
        Output:
        quadrille read method syntax and return values
        undefined return from p5.quadrille read function
        arguments for read()
        accessing out of bounds elements quadrille
        how do the coordinate system and array bounds work in p5.quadrille.js?

        User Query: "what parameters does it take?"
        History: user: how do I use the insert() method?
        Output:
        arguments for the insert method in p5.quadrille
        parameters accepted by insert()
        insert row method syntax
        p5.quadrille insert function signature
        what are the core structural methods for modifying a grid in p5.quadrille.js?

        ---
        
        USER CONVERSATION HISTORY:
        {history_text}
        
        CURRENT USER QUERY:
        {query}
        
        Output:
        """

        messages = []
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
                f.write(f'\n--- {"ORIGINAL QUERY" if i == 0 else ""}\n {q}\n\n')

        return queries

    
    def retrieve(self, multi_query, k=10):
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
    
    def get_top_results(self, db, query_embedding, top_k=4, threshold=0.0):
        similarities = []
        for chunk, chunk_embedding in db:
            similarity = self.cosine_similarity(query_embedding, chunk_embedding)
                
            if similarity >= threshold:
                similarities.append((chunk, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def load_dataset(self):
        """ Create chunks """
        # self.dataset_js = self.parse_js('../../src/quadrille.js') + self.parse_js("../../src/addon.js")
        self.dataset_js = self.parse_js('../../src/quadrille.js')
        self.dataset_md = self.parse_md('../../content/docs/')
        print(f'Loaded {len(self.dataset_js) + len(self.dataset_md)} total entries; ({len(self.dataset_js)} JS, {len(self.dataset_md)} MD)')

        # debug and analysis
        with open("js_chunks.txt", "w") as f:
            for chunk in self.dataset_js:
                f.write(f'\n\n@@@@@\n {chunk}\n\n')

        # debug and analysis
        with open("md_chunks.txt", "w") as f:
            for chunk in self.dataset_md:
                f.write(f'\n\n@@@@@\n {chunk}\n\n')

    def add_chunks_to_db(self):
        """ Create ambeddings and add them to vector db alonside their coresponsing chunk """
        for chunk in self.dataset_js:
            embedding = ollama.embed(model=self.EMBEDDING_MODEL, input=chunk)['embeddings'][0]
            self.VECTOR_DB_JS.append((chunk, embedding))
        
        for chunk in self.dataset_md:
            embedding = ollama.embed(model=self.EMBEDDING_MODEL, input=chunk, truncate=True)['embeddings'][0]
            self.VECTOR_DB_MD.append((chunk, embedding))
            
        print(f'Added {len(self.VECTOR_DB_JS)} JS chunks and {len(self.VECTOR_DB_MD)} MD chunks.')

    def cosine_similarity(self, a, b):
        """ Calculate the cosine similarity between two vectors """
        dot_product = sum([x * y for x, y in zip(a, b)])
        norm_a = sum([x ** 2 for x in a]) ** 0.5
        norm_b = sum([x ** 2 for x in b]) ** 0.5
        return dot_product / (norm_a * norm_b)
    
    def parse_md(self, folder_path):
            """ Split the MD file by headers and inject the page title into each chunk """
            chunks = []

            for root, _, files in os.walk(folder_path):
                for filename in files:
                    if not filename.endswith(".md"):
                        continue
                    
                    filepath = os.path.join(root, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()

                        title_match = re.search(r'title:\s*(.+)', content)
                        page_title = title_match.group(1).strip() if title_match else filename.replace('.md', '').replace('_', ' ')
                        
                        clean_content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL).strip()
                        
                        sections = re.split(r'\n(?=#+\s)', '\n' + clean_content)
                        
                        for section in sections:
                            section = section.strip()
                            if not section:
                                continue
                            
                            section_name = "Overview"
                            # FIX 2: Dynamically extract the name regardless of header level
                            if re.match(r'^#+', section):
                                first_line = section.split('\n')[0]
                                section_name = re.sub(r'^#+\s*', '', first_line).strip()
                            
                            chunk_text = f"DOCUMENTATION FOR: {page_title} > {section_name}\n\n{section}"
                            
                            # FIX 3: Ignore chunks that are too small to have meaning (under 50 chars)
                            if len(chunk_text) > 50:
                                chunks.append(chunk_text)

            return chunks


    def parse_js(self, filepath):
        """ Split the JS file into chunks based on JSDoc comments """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        raw_chunks = re.split(r'(?m)^(?=\s*/\*\*)', content)
        
        valid_chunks = []
        filename = os.path.basename(filepath)
        
        for chunk in raw_chunks:
            chunk = chunk.strip()
            
            if len(chunk) > 50:
                chunk_with_context = f"SOURCE CODE: {filename}\n\n{chunk}"
                valid_chunks.append(chunk_with_context)
                
        return valid_chunks


rag = RagClass()
rag.load_dataset()
rag.add_chunks_to_db()

print("The model is ready, please ask a question about p5.quadrille.js")
while True:
    q = input(f"\n\n✅ Ask {rag.LANGUAGE_MODEL}: ")
    if q == "q":
        break

    rag.ask(q)