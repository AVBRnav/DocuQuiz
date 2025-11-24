from typing import List, Dict
from src.document_loader import Document


class TextChunk:
    def __init__(self, content: str, metadata: Dict[str, any]):
        self.content = content
        self.metadata = metadata
    
    def __repr__(self):
        return f"TextChunk(source={self.metadata.get('source')}, chunk_id={self.metadata.get('chunk_id')})"


class RecursiveTextChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
    
    def chunk_documents(self, documents: List[Document]) -> List[TextChunk]:
        all_chunks = []
        
        for doc in documents:
            chunks = self._split_text(doc.content)
            
            for idx, chunk_text in enumerate(chunks):
                metadata = {
                    **doc.metadata,
                    'chunk_id': idx,
                    'total_chunks': len(chunks)
                }
                all_chunks.append(TextChunk(content=chunk_text, metadata=metadata))
        
        return all_chunks
    
    def _split_text(self, text: str) -> List[str]:
        return self._recursive_split(text, self.separators)
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break
        
        splits = text.split(separator) if separator else list(text)
        
        good_splits = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                
                if new_separators:
                    final_chunks.extend(self._recursive_split(s, new_separators))
                else:
                    final_chunks.extend(self._split_by_size(s))
        
        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)
        
        return final_chunks
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_len = len(split)
            
            if current_length + split_len + len(separator) > self.chunk_size and current_chunk:
                chunks.append(separator.join(current_chunk))
                
                overlap_chunk = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) + len(separator) <= self.chunk_overlap:
                        overlap_chunk.insert(0, s)
                        overlap_length += len(s) + len(separator)
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_length = overlap_length
            
            current_chunk.append(split)
            current_length += split_len + len(separator)
        
        if current_chunk:
            chunks.append(separator.join(current_chunk))
        
        return chunks
    
    def _split_by_size(self, text: str) -> List[str]:
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
        
        return chunks
