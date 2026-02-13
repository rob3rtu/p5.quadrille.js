from llama_cpp import Llama
from typing import List

class LocalGenerator:
    def __init__(self, model_path: str):
        print(f"Loading Llama-3 from {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1, 
            n_ctx=4096,      
            verbose=False    
        )

    def generate(self, query: str, context_results: List):
        context_block = ""
        for i, res in enumerate(context_results):
            context_block += f"--- Snippet {i+1} (Source: {res['source']}) ---\n{res['text']}\n"
        
        system_prompt = (
            "You are an expert coding assistant for the 'p5.quadrille.js' library. "
            "Your parametric memory regarding this library is unreliable. "
            "You MUST rely ONLY on the Context provided below to answer the user's question. "
            "If the Context does not contain the specific function or method requested, "
            "you must state: 'I cannot find that function in the provided documentation.' "
            "DO NOT invent function names. "
            "Use standard p5.js syntax only for the surrounding code."
        )
        
        user_prompt = (
            f"Context:\n{context_block}\n\n"
            f"User Question: {query}\n\n"
            "Answer:"
        )

        output = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1, 
            max_tokens=1024
        )
        
        return output['choices']['message']['content']