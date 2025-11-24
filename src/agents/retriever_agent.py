from typing import List, Dict, Optional
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import PineconeVectorStore


class RetrievedChunk:
    def __init__(self, text: str, score: float, metadata: Dict):
        self.text = text
        self.score = score
        self.metadata = metadata
    
    def __repr__(self):
        return f"RetrievedChunk(score={self.score:.4f}, source={self.metadata.get('filename')})"


class RetrieverAgent:
    def __init__(self, embedding_generator: EmbeddingGenerator, 
                 vector_store: PineconeVectorStore,
                 top_k: int = 5,
                 score_threshold: float = 0.7):
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.name = "retriever_agent"
    
    def retrieve(self, query: str, top_k: Optional[int] = None, 
                 filter_dict: Optional[Dict] = None) -> List[RetrievedChunk]:
        k = top_k or self.top_k
        
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=k,
            filter_dict=filter_dict,
            include_metadata=True
        )
        
        retrieved_chunks = []
        for match in results:
            if match.score >= self.score_threshold:
                chunk = RetrievedChunk(
                    text=match.metadata.get('text', ''),
                    score=match.score,
                    metadata={
                        'source': match.metadata.get('source'),
                        'filename': match.metadata.get('filename'),
                        'chunk_id': match.metadata.get('chunk_id'),
                        'id': match.id
                    }
                )
                retrieved_chunks.append(chunk)
        
        return retrieved_chunks
    
    def format_context(self, chunks: List[RetrievedChunk]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk.metadata['filename']}, "
                f"Chunk {chunk.metadata['chunk_id']}, "
                f"Relevance: {chunk.score:.2f}]\n{chunk.text}"
            )
        return "\n\n".join(context_parts)
    
    def execute(self, query: str, return_formatted: bool = False) -> Dict:
        chunks = self.retrieve(query)
        
        result = {
            'query': query,
            'chunks': chunks,
            'num_results': len(chunks),
            'context': self.format_context(chunks) if return_formatted else None
        }
        
        return result
