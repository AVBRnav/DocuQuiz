#!/usr/bin/env python3
"""
Document Ingestion Pipeline
Loads documents from the docs folder, chunks them, generates embeddings,
and stores them in Pinecone vector database.
"""

from src.config_loader import ConfigLoader
from src.document_loader import DocumentLoader
from src.text_chunker import RecursiveTextChunker
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import PineconeVectorStore


def main():
    print("=" * 60)
    print("Starting Document Ingestion Pipeline")
    print("=" * 60)
    
    config = ConfigLoader()
    
    print("\n[1/5] Loading documents...")
    loader = DocumentLoader(
        docs_folder=config.get_docs_folder(),
        supported_extensions=config.get_supported_extensions()
    )
    documents = loader.load_documents()
    print(f"Loaded {len(documents)} documents")
    
    print("\n[2/5] Chunking documents...")
    chunking_config = config.get_chunking_config()
    chunker = RecursiveTextChunker(
        chunk_size=chunking_config['chunk_size'],
        chunk_overlap=chunking_config['chunk_overlap'],
        separators=chunking_config.get('separators')
    )
    chunks = chunker.chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    print("\n[3/5] Generating embeddings...")
    openai_config = config.get_openai_config()
    embedding_gen = EmbeddingGenerator(
        api_key=openai_config['api_key'],
        model=openai_config['embedding_model']
    )
    embeddings_data = embedding_gen.generate_embeddings(chunks)
    print(f"Generated {len(embeddings_data)} embeddings")
    
    print("\n[4/5] Initializing Pinecone...")
    pinecone_config = config.get_pinecone_config()
    vector_store = PineconeVectorStore(
        api_key=pinecone_config['api_key'],
        index_name=pinecone_config['index_name'],
        dimension=pinecone_config['dimension'],
        metric=pinecone_config['metric'],
        cloud=pinecone_config['cloud'],
        region=pinecone_config['region']
    )
    vector_store.initialize_index()
    
    print("\n[5/5] Upserting embeddings to Pinecone...")
    vector_store.upsert_embeddings(embeddings_data)
    
    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    stats = vector_store.get_stats()
    print(f"Total vectors in index: {stats.total_vector_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
