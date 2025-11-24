"""
MCQ Generation Orchestrator
Coordinates the multi-agent workflow for MCQ generation.
"""

from typing import List, Dict, Any, Optional
from crewai import Crew, Task
from openai import OpenAI

from src.agents.mcq_agents import (
    MCQRetrievalAgent,
    MCQGenerationAgent,
    MCQCriticAgent,
    MCQValidationAgent
)
from src.mcq_models import MCQ, MCQGenerationResult


class MCQOrchestrator:
    """
    Orchestrates the MCQ generation workflow across multiple agents.
    Coordinates: Retrieval → Generation → Critique → Validation
    """
    
    def __init__(
        self,
        retrieval_agent: MCQRetrievalAgent,
        openai_api_key: str,
        generation_model: str = "gpt-4",
        critic_model: str = "gpt-4",
        generation_temperature: float = 0.7,
        critic_temperature: float = 0.3
    ):
        """
        Initialize the orchestrator with all required agents.
        
        Args:
            retrieval_agent: Configured MCQ retrieval agent
            openai_api_key: OpenAI API key for LLM agents
            generation_model: Model for MCQ generation
            critic_model: Model for critique
            generation_temperature: Temperature for generation
            critic_temperature: Temperature for critique
        """
        self.retrieval_agent = retrieval_agent
        
        # Initialize OpenAI client
        llm_client = OpenAI(api_key=openai_api_key)
        
        # Initialize specialized agents
        self.generation_agent = MCQGenerationAgent(
            llm_client=llm_client,
            model=generation_model,
            temperature=generation_temperature
        )
        
        self.critic_agent = MCQCriticAgent(
            llm_client=llm_client,
            model=critic_model,
            temperature=critic_temperature
        )
        
        self.validation_agent = MCQValidationAgent()
        
        print("MCQ Orchestrator initialized with all agents")
    
    def generate_mcqs(
        self,
        query: str,
        num_mcqs: int = 5,
        difficulty: Optional[str] = None,
        top_k_chunks: int = 5,
        min_quality_score: float = 7.0
    ) -> MCQGenerationResult:
        """
        Execute the complete MCQ generation workflow.
        
        Args:
            query: Topic or query for MCQ generation
            num_mcqs: Number of MCQs to generate
            difficulty: Optional difficulty level (easy/medium/hard)
            top_k_chunks: Number of context chunks to retrieve
            min_quality_score: Minimum quality score for MCQs (0-10)
            
        Returns:
            MCQGenerationResult with MCQs, critiques, and validations
        """
        print("\n" + "=" * 70)
        print("MCQ GENERATION WORKFLOW")
        print("=" * 70)
        
        # Step 1: Retrieval
        print(f"\n[Step 1/4] Retrieving context for: '{query}'")
        retrieval_result = self.retrieval_agent.retrieve_context(
            query=query,
            top_k=top_k_chunks
        )
        
        context_chunks = retrieval_result['chunks']
        print(f"✓ Retrieved {len(context_chunks)} relevant chunks")
        
        if not context_chunks:
            print("✗ No relevant context found. Cannot generate MCQs.")
            return MCQGenerationResult(
                query=query,
                mcqs=[],
                critiques=[],
                validations=[]
            )
        
        # Step 2: Generation
        print(f"\n[Step 2/4] Generating {num_mcqs} MCQs...")
        mcqs = self.generation_agent.generate_mcqs(
            context_chunks=context_chunks,
            num_mcqs=num_mcqs,
            difficulty=difficulty
        )
        print(f"✓ Generated {len(mcqs)} MCQs")
        
        if not mcqs:
            print("✗ MCQ generation failed.")
            return MCQGenerationResult(
                query=query,
                mcqs=[],
                critiques=[],
                validations=[]
            )
        
        # Step 3: Critique
        print(f"\n[Step 3/4] Critiquing {len(mcqs)} MCQs...")
        critiques = self.critic_agent.critique_mcqs(
            mcqs=mcqs,
            context_chunks=context_chunks
        )
        print(f"✓ Completed critique for {len(critiques)} MCQs")
        
        # Display critique summary
        for i, critique in enumerate(critiques):
            print(f"  MCQ {i+1}: Overall Score = {critique.overall_score:.1f}/10")
        
        # Step 4: Validation
        print(f"\n[Step 4/4] Validating {len(mcqs)} MCQs...")
        validations = self.validation_agent.validate_mcqs(
            mcqs=mcqs,
            critiques=critiques,
            context_chunks=context_chunks
        )
        
        valid_count = sum(1 for v in validations if v.is_valid())
        print(f"✓ Validation complete: {valid_count}/{len(mcqs)} MCQs passed")
        
        # Create final result
        result = MCQGenerationResult(
            query=query,
            mcqs=mcqs,
            critiques=critiques,
            validations=validations
        )
        
        # Summary
        print("\n" + "=" * 70)
        print("GENERATION SUMMARY")
        print("=" * 70)
        print(f"Query: {query}")
        print(f"Total MCQs Generated: {len(mcqs)}")
        print(f"Valid MCQs: {len(result.valid_mcqs)}")
        print(f"Invalid MCQs: {len(result.invalid_mcqs)}")
        
        avg_score = sum(c.overall_score for c in critiques) / len(critiques) if critiques else 0
        print(f"Average Quality Score: {avg_score:.2f}/10")
        print("=" * 70)
        
        return result
    
    def generate_mcqs_batch(
        self,
        queries: List[str],
        num_mcqs_per_query: int = 3,
        **kwargs
    ) -> List[MCQGenerationResult]:
        """
        Generate MCQs for multiple queries in batch.
        
        Args:
            queries: List of queries/topics
            num_mcqs_per_query: MCQs to generate per query
            **kwargs: Additional arguments for generate_mcqs
            
        Returns:
            List of MCQGenerationResult objects
        """
        results = []
        
        print(f"\nBatch MCQ Generation: {len(queries)} queries")
        print("=" * 70)
        
        for i, query in enumerate(queries, 1):
            print(f"\n[Query {i}/{len(queries)}]")
            result = self.generate_mcqs(
                query=query,
                num_mcqs=num_mcqs_per_query,
                **kwargs
            )
            results.append(result)
        
        # Batch summary
        total_valid = sum(len(r.valid_mcqs) for r in results)
        total_generated = sum(len(r.mcqs) for r in results)
        
        print("\n" + "=" * 70)
        print("BATCH SUMMARY")
        print("=" * 70)
        print(f"Queries Processed: {len(queries)}")
        print(f"Total MCQs Generated: {total_generated}")
        print(f"Total Valid MCQs: {total_valid}")
        print(f"Success Rate: {(total_valid/total_generated*100):.1f}%")
        print("=" * 70)
        
        return results
    
    def refine_mcq(
        self,
        mcq: MCQ,
        critique: Any,
        context_chunk: Dict[str, Any]
    ) -> MCQ:
        """
        Refine an MCQ based on critique feedback.
        (Future enhancement for iterative improvement)
        
        Args:
            mcq: Original MCQ
            critique: Critique result
            context_chunk: Source context
            
        Returns:
            Refined MCQ
        """
        # TODO: Implement iterative refinement
        # This would send the MCQ + critique back to the generation agent
        # for revision based on specific suggestions
        return mcq

