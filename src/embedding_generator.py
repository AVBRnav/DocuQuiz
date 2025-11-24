from typing import List
from openai import OpenAI
from src.text_chunker import TextChunk


class EmbeddingGenerator:
    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate_embeddings(self, chunks: List[TextChunk]) -> List[dict]:
        embeddings_data = []
        
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_texts = [chunk.content for chunk in batch]
            
            response = self.client.embeddings.create(
                input=batch_texts,
                model=self.model
            )
            
            for idx, embedding_obj in enumerate(response.data):
                chunk = batch[idx]
                embeddings_data.append({
                    'id': f"{chunk.metadata['source']}_{chunk.metadata['chunk_id']}",
                    'values': embedding_obj.embedding,
                    'metadata': {
                        'text': chunk.content,
                        'source': chunk.metadata['source'],
                        'filename': chunk.metadata['filename'],
                        'chunk_id': chunk.metadata['chunk_id'],
                        'total_chunks': chunk.metadata['total_chunks']
                    }
                })
        
        return embeddings_data
    
    def generate_query_embedding(self, query: str) -> List[float]:
        response = self.client.embeddings.create(
            input=[query],
            model=self.model
        )
        return response.data[0].embedding
