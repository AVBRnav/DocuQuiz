#!/usr/bin/env python3
"""
Agentic MCQ Generation Pipeline
End-to-end workflow for generating MCQs using multi-agent architecture.
"""

import sys
import json
from pathlib import Path
from typing import Optional

from src.config_loader import ConfigLoader
from src.embedding_generator import EmbeddingGenerator
from src.vector_store import PineconeVectorStore
from src.agents.retriever_agent import RetrieverAgent
from src.agents.mcq_agents import MCQRetrievalAgent
from src.agents.mcq_orchestrator import MCQOrchestrator


class AgenticMCQPipeline:
    """
    Main pipeline for agentic MCQ generation workflow.
    Integrates with existing RAG infrastructure.
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize the agentic MCQ pipeline.
        
        Args:
            config: ConfigLoader instance with system configuration
        """
        self.config = config
        self.orchestrator = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize all components of the MCQ pipeline"""
        print("=" * 70)
        print("INITIALIZING AGENTIC MCQ GENERATION PIPELINE")
        print("=" * 70)
        
        # Get configuration
        openai_config = self.config.get_openai_config()
        pinecone_config = self.config.get_pinecone_config()
        retrieval_config = self.config.get_retrieval_config()
        
        print("\n[1/4] Initializing Embedding Generator...")
        embedding_gen = EmbeddingGenerator(
            api_key=openai_config['api_key'],
            model=openai_config['embedding_model']
        )
        print("✓ Embedding generator ready")
        
        print("\n[2/4] Connecting to Pinecone Vector Store...")
        vector_store = PineconeVectorStore(
            api_key=pinecone_config['api_key'],
            index_name=pinecone_config['index_name'],
            dimension=pinecone_config['dimension'],
            metric=pinecone_config['metric'],
            cloud=pinecone_config['cloud'],
            region=pinecone_config['region']
        )
        vector_store.initialize_index()
        print("✓ Vector store connected")
        
        print("\n[3/4] Setting up Retrieval Agent...")
        base_retriever = RetrieverAgent(
            embedding_generator=embedding_gen,
            vector_store=vector_store,
            top_k=retrieval_config['top_k'],
            score_threshold=retrieval_config['score_threshold']
        )
        
        mcq_retrieval_agent = MCQRetrievalAgent(base_retriever=base_retriever)
        print("✓ Retrieval agent ready")
        
        print("\n[4/4] Initializing MCQ Orchestrator...")
        self.orchestrator = MCQOrchestrator(
            retrieval_agent=mcq_retrieval_agent,
            openai_api_key=openai_config['api_key'],
            generation_model=self.config.get('agents.reasoning.model', 'gpt-4'),
            generation_temperature=self.config.get('agents.reasoning.temperature', 0.7)
        )
        print("✓ Orchestrator ready")
        
        print("\n" + "=" * 70)
        print("PIPELINE INITIALIZATION COMPLETE")
        print("=" * 70)
    
    def generate_mcqs(
        self,
        query: str,
        num_mcqs: int = 5,
        difficulty: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> dict:
        """
        Generate MCQs for a given query/topic.
        
        Args:
            query: Topic or question for MCQ generation
            num_mcqs: Number of MCQs to generate (default: 5)
            difficulty: Optional difficulty level (easy/medium/hard)
            output_file: Optional path to save results as JSON
            
        Returns:
            Dictionary with generation results
        """
        result = self.orchestrator.generate_mcqs(
            query=query,
            num_mcqs=num_mcqs,
            difficulty=difficulty
        )
        
        # Save to file if requested
        if output_file:
            self._save_results(result, output_file)
        
        return result.to_dict()
    
    def generate_mcqs_batch(
        self,
        queries: list,
        num_mcqs_per_query: int = 3,
        difficulty: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> list:
        """
        Generate MCQs for multiple queries in batch.
        
        Args:
            queries: List of topics/questions
            num_mcqs_per_query: MCQs per query (default: 3)
            difficulty: Optional difficulty level
            output_file: Optional path to save results
            
        Returns:
            List of result dictionaries
        """
        results = self.orchestrator.generate_mcqs_batch(
            queries=queries,
            num_mcqs_per_query=num_mcqs_per_query,
            difficulty=difficulty
        )
        
        results_dict = [r.to_dict() for r in results]
        
        # Save to file if requested
        if output_file:
            self._save_batch_results(results_dict, output_file)
        
        return results_dict
    
    def _save_results(self, result, output_file: str):
        """Save MCQ generation results to JSON file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to: {output_file}")
    
    def _save_batch_results(self, results: list, output_file: str):
        """Save batch results to JSON file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Batch results saved to: {output_file}")
    
    def display_mcqs(self, result_dict: dict, show_invalid: bool = False):
        """
        Display MCQs in a readable format.
        
        Args:
            result_dict: Result dictionary from generate_mcqs
            show_invalid: Whether to show invalid MCQs
        """
        print("\n" + "=" * 70)
        print("GENERATED MCQs")
        print("=" * 70)
        
        mcqs_to_show = result_dict['valid_mcqs']
        if show_invalid:
            mcqs_to_show = result_dict['mcqs']
        
        for i, mcq in enumerate(mcqs_to_show, 1):
            print(f"\n{'─' * 70}")
            print(f"MCQ #{i}")
            print(f"{'─' * 70}")
            print(f"Question: {mcq['question']}")
            print(f"\nOptions:")
            for opt in mcq['options']:
                marker = "✓" if opt['is_correct'] else " "
                print(f"  [{marker}] {opt['label']}. {opt['text']}")
            print(f"\nCorrect Answer: {mcq['correct_answer']}")
            print(f"Explanation: {mcq['explanation']}")
            print(f"\nMetadata:")
            print(f"  • Difficulty: {mcq['difficulty']}")
            print(f"  • Source: {mcq['source_filename']}")
            print(f"  • Chunk ID: {mcq['chunk_id']}")
        
        print("\n" + "=" * 70)


def interactive_mode(pipeline: AgenticMCQPipeline):
    """Run pipeline in interactive mode"""
    print("\n" + "=" * 70)
    print("INTERACTIVE MCQ GENERATION MODE")
    print("=" * 70)
    print("\nCommands:")
    print("  • Enter a topic/query to generate MCQs")
    print("  • Type 'batch' to enter batch mode")
    print("  • Type 'exit' or 'quit' to exit")
    print("=" * 70)
    
    while True:
        try:
            query = input("\nEnter topic or query: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break
            
            if query.lower() == 'batch':
                batch_mode(pipeline)
                continue
            
            if not query:
                continue
            
            # Get parameters
            try:
                num_str = input("Number of MCQs (default 5): ").strip()
                num_mcqs = int(num_str) if num_str else 5
            except ValueError:
                num_mcqs = 5
            
            difficulty_input = input("Difficulty (easy/medium/hard, or press Enter): ").strip()
            difficulty = difficulty_input if difficulty_input else None
            
            save_option = input("Save to file? (y/n, default n): ").strip().lower()
            output_file = None
            if save_option == 'y':
                output_file = input("Output filename (e.g., mcqs.json): ").strip()
                if not output_file:
                    output_file = "output/mcqs.json"
            
            # Generate MCQs
            print()
            result = pipeline.generate_mcqs(
                query=query,
                num_mcqs=num_mcqs,
                difficulty=difficulty,
                output_file=output_file
            )
            
            # Display results
            pipeline.display_mcqs(result)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


def batch_mode(pipeline: AgenticMCQPipeline):
    """Run pipeline in batch mode"""
    print("\n" + "=" * 70)
    print("BATCH MCQ GENERATION MODE")
    print("=" * 70)
    print("Enter queries one per line. Type 'done' when finished.")
    print("=" * 70)
    
    queries = []
    while True:
        query = input(f"Query {len(queries)+1}: ").strip()
        if query.lower() == 'done':
            break
        if query:
            queries.append(query)
    
    if not queries:
        print("No queries entered.")
        return
    
    try:
        num_str = input(f"\nMCQs per query (default 3): ").strip()
        num_mcqs = int(num_str) if num_str else 3
    except ValueError:
        num_mcqs = 3
    
    difficulty_input = input("Difficulty (easy/medium/hard, or press Enter): ").strip()
    difficulty = difficulty_input if difficulty_input else None
    
    output_file = input("Output filename (default: output/batch_mcqs.json): ").strip()
    if not output_file:
        output_file = "output/batch_mcqs.json"
    
    # Generate batch
    print()
    results = pipeline.generate_mcqs_batch(
        queries=queries,
        num_mcqs_per_query=num_mcqs,
        difficulty=difficulty,
        output_file=output_file
    )
    
    print(f"\n✓ Generated MCQs for {len(results)} queries")


def main():
    """Main entry point"""
    config = ConfigLoader()
    pipeline = AgenticMCQPipeline(config)
    
    if len(sys.argv) > 1:
        # Command-line mode
        query = ' '.join(sys.argv[1:])
        result = pipeline.generate_mcqs(query=query, num_mcqs=5)
        pipeline.display_mcqs(result)
    else:
        # Interactive mode
        interactive_mode(pipeline)


if __name__ == "__main__":
    main()

