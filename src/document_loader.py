from pathlib import Path
from typing import List, Dict
import PyPDF2


class Document:
    def __init__(self, content: str, metadata: Dict[str, str]):
        self.content = content
        self.metadata = metadata
    
    def __repr__(self):
        return f"Document(source={self.metadata.get('source')}, length={len(self.content)})"


class DocumentLoader:
    def __init__(self, docs_folder: str, supported_extensions: List[str]):
        self.docs_folder = Path(docs_folder)
        self.supported_extensions = supported_extensions
    
    def load_documents(self) -> List[Document]:
        if not self.docs_folder.exists():
            raise FileNotFoundError(f"Documents folder not found: {self.docs_folder}")
        
        documents = []
        
        for file_path in self.docs_folder.rglob('*'):
            if file_path.is_file() and file_path.suffix in self.supported_extensions:
                doc = self._load_file(file_path)
                if doc:
                    documents.append(doc)
        
        return documents
    
    def _load_file(self, file_path: Path) -> Document:
        try:
            if file_path.suffix == '.pdf':
                content = self._load_pdf(file_path)
            else:
                content = self._load_text(file_path)
            
            metadata = {
                'source': str(file_path),
                'filename': file_path.name,
                'extension': file_path.suffix
            }
            
            return Document(content=content, metadata=metadata)
        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def _load_text(self, file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_pdf(self, file_path: Path) -> str:
        text = []
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text.append(page.extract_text())
        return '\n'.join(text)
