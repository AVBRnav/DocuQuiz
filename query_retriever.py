#!/usr/bin/env python3
"""
Query Retriever
Interactive script to query the RAG system and retrieve relevant chunks.
"""

import sys
from src.config_loader import ConfigLoader
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import PineconeVectorStore
from src.agents.retriever_agent import RetrieverAgent


def main():
    config = ConfigLoader()
    
    print("Initializing Retriever Agent...")
    
    openai_config = config.get_openai_config()
    embedding_gen = EmbeddingGenerator(
        api_key=openai_config['api_key'],
        model=openai_config['embedding_model']
    )
    
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
    
    retrieval_config = config.get_retrieval_config()
    retriever = RetrieverAgent(
        embedding_generator=embedding_gen,
        vector_store=vector_store,
        top_k=retrieval_config['top_k'],
        score_threshold=retrieval_config['score_threshold']
    )
    
    print("\nRetriever Agent Ready!")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
        process_query(retriever, query)
    else:
        interactive_mode(retriever)


def process_query(retriever: RetrieverAgent, query: str):
    print(f"\nQuery: {query}")
    print("-" * 60)
    
    result = retriever.execute(query, return_formatted=True)
    
    print(f"\nFound {result['num_results']} relevant chunks:\n")
    
    for i, chunk in enumerate(result['chunks'], 1):
        print(f"[{i}] Score: {chunk.score:.4f} | Source: {chunk.metadata['filename']}")
        print(f"    {chunk.text[:200]}...")
        print()
    
    if result['context']:
        print("\n" + "=" * 60)
        print("FORMATTED CONTEXT FOR LLM:")
        print("=" * 60)
        print(result['context'])


def interactive_mode(retriever: RetrieverAgent):
    print("Interactive Query Mode (type 'exit' to quit)")
    print("=" * 60)
    
    while True:
        try:
            query = input("\nEnter your query: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            process_query(retriever, query)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
