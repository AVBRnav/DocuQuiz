from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec
import time


class PineconeVectorStore:
    def __init__(self, api_key: str, index_name: str, dimension: int, 
                 metric: str = "cosine", cloud: str = "aws", region: str = "us-east-1"):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self.index = None
    
    def initialize_index(self):
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud=self.cloud,
                    region=self.region
                )
            )
            
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            
            print(f"Index {self.index_name} created successfully")
        else:
            print(f"Using existing index: {self.index_name}")
        
        self.index = self.pc.Index(self.index_name)
    
    def upsert_embeddings(self, embeddings_data: List[Dict], batch_size: int = 100):
        if not self.index:
            raise ValueError("Index not initialized. Call initialize_index() first.")
        
        total = len(embeddings_data)
        for i in range(0, total, batch_size):
            batch = embeddings_data[i:i + batch_size]
            self.index.upsert(vectors=batch)
            print(f"Upserted {min(i + batch_size, total)}/{total} embeddings")
    
    def query(self, query_embedding: List[float], top_k: int = 5, 
              filter_dict: Dict = None, include_metadata: bool = True) -> List[Dict]:
        if not self.index:
            raise ValueError("Index not initialized. Call initialize_index() first.")
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=include_metadata
        )
        
        return results.matches
    
    def delete_all(self):
        if not self.index:
            raise ValueError("Index not initialized. Call initialize_index() first.")
        
        self.index.delete(delete_all=True)
        print(f"Deleted all vectors from index: {self.index_name}")
    
    def get_stats(self):
        if not self.index:
            raise ValueError("Index not initialized. Call initialize_index() first.")
        
        return self.index.describe_index_stats()
