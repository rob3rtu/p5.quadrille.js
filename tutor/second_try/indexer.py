import os
import re
import glob
from typing import List

class Indexer:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.chunks = list()

    def load_and_chunk(self) -> List:
        """Loads the JS and MD files."""
        print(f"Scanning repository")
        
        js_files = glob.glob(os.path.join(self.repo_path, "**/*.js"))
        for f in js_files:
            self._parse_js_file(f)

        md_files = glob.glob(os.path.join(self.repo_path, "**/*.md"), recursive=True)
        for f in md_files:
            self._parse_md_file(f)
            
        print(f"Total semantic chunks generated: {len(self.chunks)}")
        return self.chunks

    def _parse_js_file(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Regex to capture JSDoc comments + function signature
        # This explicitly binds the documentation to the code.
        pattern = re.compile(r'(/\*\**?\*/\s*[\w\.]*\s*function\s+\w+\s*\(.*?\))', re.MULTILINE)
        
        matches = pattern.finditer(content)
        for match in matches:
            chunk_text = match.group(1)
            full_chunk = f"File: {os.path.basename(filepath)}\nLanguage: JavaScript\nCode Definition:\n{chunk_text} {{... }}"
            
            self.chunks.append({
                "text": full_chunk,
                "source": filepath,
                "type": "code"
            })

    def _parse_md_file(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by Headers (H1, H2, H3) to respect document structure
        sections = re.split(r'(^#+\s.*)', content, flags=re.MULTILINE)
        
        for i in range(1, len(sections), 2):
            if i+1 < len(sections):
                header = sections[i].strip()
                body = sections[i+1].strip()
                self.chunks.append({
                    "text": f"Source: {os.path.basename(filepath)}\nSection: {header}\nContent:\n{body}",
                    "source": filepath,
                    "type": "documentation"
                })