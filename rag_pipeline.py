#!/usr/bin/env python3
"""
Full RAG Pipeline with Retriever + Reasoning Agent
Demonstrates the complete agentic RAG workflow.
"""

import sys
from src.config_loader import ConfigLoader
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import PineconeVectorStore
from src.agents.retriever_agent import RetrieverAgent
from src.agents.reasoning_agent import ReasoningAgent


class AgenticRAGPipeline:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.retriever = None
        self.reasoner = None
        self._initialize_agents()
    
    def _initialize_agents(self):
        print("Initializing Agentic RAG Pipeline...")
        
        openai_config = self.config.get_openai_config()
        embedding_gen = EmbeddingGenerator(
            api_key=openai_config['api_key'],
            model=openai_config['embedding_model']
        )
        
        pinecone_config = self.config.get_pinecone_config()
        vector_store = PineconeVectorStore(
            api_key=pinecone_config['api_key'],
            index_name=pinecone_config['index_name'],
            dimension=pinecone_config['dimension'],
            metric=pinecone_config['metric'],
            cloud=pinecone_config['cloud'],
            region=pinecone_config['region']
        )
        vector_store.initialize_index()
        
        retrieval_config = self.config.get_retrieval_config()
        self.retriever = RetrieverAgent(
            embedding_generator=embedding_gen,
            vector_store=vector_store,
            top_k=retrieval_config['top_k'],
            score_threshold=retrieval_config['score_threshold']
        )
        
        agent_config = self.config.get('agents.reasoning', {})
        self.reasoner = ReasoningAgent(
            api_key=openai_config['api_key'],
            model=agent_config.get('model', 'gpt-4'),
            temperature=agent_config.get('temperature', 0.7),
            max_tokens=agent_config.get('max_tokens', 2000)
        )
        
        print("Pipeline Ready!\n")
    
    def query(self, question: str, verbose: bool = True) -> dict:
        if verbose:
            print(f"Question: {question}")
            print("=" * 60)
            print(f"\n[Agent: {self.retriever.name}] Retrieving relevant chunks...")
        
        retrieval_result = self.retriever.execute(question)
        chunks = retrieval_result['chunks']
        
        if verbose:
            print(f"Retrieved {len(chunks)} chunks")
            for i, chunk in enumerate(chunks, 1):
                print(f"  [{i}] {chunk.metadata['filename']} (score: {chunk.score:.4f})")
        
        if not chunks:
            return {
                'question': question,
                'answer': "No relevant information found in the knowledge base.",
                'sources': []
            }
        
        if verbose:
            print(f"\n[Agent: {self.reasoner.name}] Generating response...")
        
        system_prompt = self.config.get('prompts.system_prompt')
        reasoning_result = self.reasoner.execute(
            query=question,
            retrieved_chunks=chunks,
            system_prompt=system_prompt
        )
        
        result = {
            'question': question,
            'answer': reasoning_result['response'],
            'sources': reasoning_result['sources'],
            'num_sources': reasoning_result['num_sources']
        }
        
        if verbose:
            print("\n" + "=" * 60)
            print("ANSWER:")
            print("=" * 60)
            print(result['answer'])
            print("\n" + "=" * 60)
            print(f"Sources: {result['num_sources']} documents")
            print("=" * 60)
        
        return result


def main():
    config = ConfigLoader()
    pipeline = AgenticRAGPipeline(config)
    
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
        pipeline.query(query)
    else:
        interactive_mode(pipeline)


def interactive_mode(pipeline: AgenticRAGPipeline):
    print("Interactive RAG Mode (type 'exit' to quit)")
    print("=" * 60)
    
    while True:
        try:
            query = input("\nAsk a question: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            print()
            pipeline.query(query)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
