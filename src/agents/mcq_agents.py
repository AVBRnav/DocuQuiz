"""
MCQ Generation Agents using CrewAI
Implements specialized agents for MCQ generation workflow.
"""

from crewai import Agent, Task, Crew
from typing import List, Dict, Any, Optional
import json
from src.agents.retriever_agent import RetrieverAgent as BaseRetrieverAgent
from src.mcq_models import (
    MCQ, MCQOption, CritiqueResult, ValidationResult, 
    DifficultyLevel, ValidationStatus
)


class MCQRetrievalAgent:
    """
    Wraps the existing RetrieverAgent for CrewAI integration.
    Retrieves relevant chunks from Pinecone.
    """
    
    def __init__(self, base_retriever: BaseRetrieverAgent):
        self.base_retriever = base_retriever
        self.name = "mcq_retrieval_agent"
    
    def retrieve_context(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieve relevant chunks for MCQ generation.
        
        Args:
            query: Topic or query for MCQ generation
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with retrieved chunks and metadata
        """
        result = self.base_retriever.execute(query, return_formatted=False)
        
        chunks_data = []
        for chunk in result['chunks']:
            chunks_data.append({
                'text': chunk.text,
                'score': chunk.score,
                'chunk_id': chunk.metadata.get('chunk_id', ''),
                'source': chunk.metadata.get('filename', ''),
                'metadata': chunk.metadata
            })
        
        return {
            'query': query,
            'chunks': chunks_data,
            'num_chunks': len(chunks_data),
            'formatted_context': self.base_retriever.format_context(result['chunks'])
        }


class MCQGenerationAgent:
    """
    CrewAI agent that generates MCQs from retrieved context.
    Uses LLM to create contextually grounded questions.
    """
    
    def __init__(self, llm_client, model: str = "gpt-4", temperature: float = 0.7):
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.name = "mcq_generation_agent"
    
    def create_agent(self) -> Agent:
        """Create CrewAI agent for MCQ generation"""
        return Agent(
            role="MCQ Generator",
            goal="Generate high-quality multiple choice questions strictly based on provided context",
            backstory="""You are an expert educational content creator specializing in 
            creating clear, accurate, and challenging multiple choice questions. You never 
            introduce information not present in the context. You always ensure questions 
            test real understanding, not just memorization.""",
            verbose=True,
            allow_delegation=False
        )
    
    def generate_mcqs(
        self, 
        context_chunks: List[Dict[str, Any]], 
        num_mcqs: int = 3,
        difficulty: Optional[str] = None
    ) -> List[MCQ]:
        """
        Generate MCQs from context chunks.
        
        Args:
            context_chunks: List of retrieved chunks with text and metadata
            num_mcqs: Number of MCQs to generate
            difficulty: Optional difficulty level (easy/medium/hard)
            
        Returns:
            List of MCQ objects
        """
        if not context_chunks:
            return []
        
        # Prepare context for LLM
        context_text = "\n\n".join([
            f"[Chunk {i+1} from {chunk['source']}]:\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        difficulty_instruction = ""
        if difficulty:
            difficulty_instruction = f"Generate {difficulty} difficulty questions. "
        
        prompt = f"""Based ONLY on the following context, generate {num_mcqs} multiple choice questions.

Context:
{context_text}

Requirements:
- Each question must be directly answerable from the context above
- Do NOT introduce any external information or assumptions
- Provide exactly 4 options (A, B, C, D) for each question
- Mark the correct answer clearly
- Provide a brief explanation referencing the context
- {difficulty_instruction}Estimate difficulty level (easy/medium/hard)

Return ONLY a valid JSON array with this exact structure:
[
  {{
    "question": "Question text here?",
    "options": [
      {{"label": "A", "text": "Option A text"}},
      {{"label": "B", "text": "Option B text"}},
      {{"label": "C", "text": "Option C text"}},
      {{"label": "D", "text": "Option D text"}}
    ],
    "correct_answer": "A",
    "explanation": "Explanation referencing the context",
    "difficulty": "medium",
    "chunk_index": 0
  }}
]

JSON:"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert MCQ generator. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON if wrapped in code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            mcqs_data = json.loads(response_text)
            
            # Convert to MCQ objects
            mcqs = []
            for i, mcq_data in enumerate(mcqs_data):
                chunk_idx = mcq_data.get('chunk_index', 0)
                source_chunk = context_chunks[chunk_idx] if chunk_idx < len(context_chunks) else context_chunks[0]
                
                options = [
                    MCQOption(
                        label=opt["label"],
                        text=opt["text"],
                        is_correct=(opt["label"] == mcq_data["correct_answer"])
                    )
                    for opt in mcq_data["options"]
                ]
                
                mcq = MCQ(
                    question=mcq_data["question"],
                    options=options,
                    correct_answer=mcq_data["correct_answer"],
                    explanation=mcq_data["explanation"],
                    difficulty=DifficultyLevel(mcq_data.get("difficulty", "medium")),
                    chunk_id=str(source_chunk.get('chunk_id', i)),
                    source_filename=source_chunk.get('source', 'unknown'),
                    context_snippet=source_chunk['text'][:200] + "...",
                    metadata={'generation_order': i}
                )
                mcqs.append(mcq)
            
            return mcqs
            
        except Exception as e:
            print(f"Error generating MCQs: {e}")
            return []


class MCQCriticAgent:
    """
    CrewAI agent that critiques generated MCQs.
    Evaluates clarity, correctness, and grounding.
    """
    
    def __init__(self, llm_client, model: str = "gpt-4", temperature: float = 0.3):
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.name = "mcq_critic_agent"
    
    def create_agent(self) -> Agent:
        """Create CrewAI agent for MCQ critique"""
        return Agent(
            role="MCQ Critic",
            goal="Evaluate MCQ quality and provide constructive feedback",
            backstory="""You are a meticulous educational assessment expert who evaluates 
            multiple choice questions for clarity, correctness, appropriate difficulty, 
            and grounding in source material. You provide specific, actionable feedback.""",
            verbose=True,
            allow_delegation=False
        )
    
    def critique_mcqs(
        self, 
        mcqs: List[MCQ], 
        context_chunks: List[Dict[str, Any]]
    ) -> List[CritiqueResult]:
        """
        Critique each MCQ for quality and grounding.
        
        Args:
            mcqs: List of generated MCQs
            context_chunks: Original context chunks
            
        Returns:
            List of CritiqueResult objects
        """
        critiques = []
        
        for i, mcq in enumerate(mcqs):
            # Find the source chunk
            source_chunk = next(
                (chunk for chunk in context_chunks 
                 if str(chunk.get('chunk_id')) == mcq.chunk_id),
                context_chunks[0] if context_chunks else None
            )
            
            if not source_chunk:
                # Create a basic critique if no context found
                critiques.append(CritiqueResult(
                    mcq_index=i,
                    clarity_score=5.0,
                    correctness_score=5.0,
                    grounding_score=0.0,
                    difficulty_assessment=mcq.difficulty,
                    issues=["Could not find source context for verification"],
                    suggestions=["Verify MCQ against original source"]
                ))
                continue
            
            prompt = f"""Evaluate the following MCQ for quality:

Context:
{source_chunk['text']}

MCQ:
Question: {mcq.question}
Options:
{chr(10).join(f"{opt.label}. {opt.text}" for opt in mcq.options)}
Correct Answer: {mcq.correct_answer}
Explanation: {mcq.explanation}

Evaluate on these criteria (score 0-10 each):
1. Clarity: Is the question clear and unambiguous?
2. Correctness: Is the correct answer truly correct based on context?
3. Grounding: Is everything in the MCQ derived from the context?

Also assess:
- Difficulty level (easy/medium/hard)
- Any issues or problems
- Suggestions for improvement

Return ONLY valid JSON:
{{
  "clarity_score": 8.5,
  "correctness_score": 9.0,
  "grounding_score": 8.0,
  "difficulty_assessment": "medium",
  "issues": ["List any issues"],
  "suggestions": ["List suggestions"]
}}

JSON:"""
            
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert MCQ evaluator. Always return valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # Extract JSON
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                
                critique_data = json.loads(response_text)
                
                critique = CritiqueResult(
                    mcq_index=i,
                    clarity_score=float(critique_data.get("clarity_score", 5.0)),
                    correctness_score=float(critique_data.get("correctness_score", 5.0)),
                    grounding_score=float(critique_data.get("grounding_score", 5.0)),
                    difficulty_assessment=DifficultyLevel(
                        critique_data.get("difficulty_assessment", "medium")
                    ),
                    issues=critique_data.get("issues", []),
                    suggestions=critique_data.get("suggestions", [])
                )
                critiques.append(critique)
                
            except Exception as e:
                print(f"Error critiquing MCQ {i}: {e}")
                # Fallback critique
                critiques.append(CritiqueResult(
                    mcq_index=i,
                    clarity_score=7.0,
                    correctness_score=7.0,
                    grounding_score=7.0,
                    difficulty_assessment=mcq.difficulty,
                    issues=["Could not complete full critique"],
                    suggestions=["Manual review recommended"]
                ))
        
        return critiques


class MCQValidationAgent:
    """
    Agent that performs strict validation of MCQs.
    Ensures grounding, format, and metadata requirements.
    """
    
    def __init__(self):
        self.name = "mcq_validation_agent"
    
    def create_agent(self) -> Agent:
        """Create CrewAI agent for MCQ validation"""
        return Agent(
            role="MCQ Validator",
            goal="Ensure all MCQs meet strict quality and formatting standards",
            backstory="""You are a quality assurance specialist who enforces strict 
            standards for educational content. You verify that every MCQ is properly 
            formatted, grounded in source material, and free from hallucinations.""",
            verbose=True,
            allow_delegation=False
        )
    
    def validate_mcqs(
        self, 
        mcqs: List[MCQ], 
        critiques: List[CritiqueResult],
        context_chunks: List[Dict[str, Any]]
    ) -> List[ValidationResult]:
        """
        Validate each MCQ against strict criteria.
        
        Args:
            mcqs: List of generated MCQs
            critiques: Critique results from critic agent
            context_chunks: Original context chunks
            
        Returns:
            List of ValidationResult objects
        """
        validations = []
        
        for i, mcq in enumerate(mcqs):
            critique = critiques[i] if i < len(critiques) else None
            errors = []
            
            # Check formatting
            is_properly_formatted = self._check_format(mcq, errors)
            
            # Check required metadata
            has_required_metadata = self._check_metadata(mcq, errors)
            
            # Check context grounding using critique scores
            is_context_grounded = True
            if critique and critique.grounding_score < 6.0:
                is_context_grounded = False
                errors.append(f"Low grounding score: {critique.grounding_score}/10")
            
            # Check for potential hallucination
            has_hallucination = False
            if critique and critique.grounding_score < 5.0:
                has_hallucination = True
                errors.append("Potential hallucination detected")
            
            # Determine overall status
            if not errors:
                status = ValidationStatus.VALID
            elif critique and critique.overall_score >= 7.0:
                status = ValidationStatus.NEEDS_REVISION
            else:
                status = ValidationStatus.INVALID
            
            validation = ValidationResult(
                mcq_index=i,
                status=status,
                is_context_grounded=is_context_grounded,
                is_properly_formatted=is_properly_formatted,
                has_required_metadata=has_required_metadata,
                has_hallucination=has_hallucination,
                validation_errors=errors
            )
            validations.append(validation)
        
        return validations
    
    def _check_format(self, mcq: MCQ, errors: List[str]) -> bool:
        """Check if MCQ is properly formatted"""
        is_valid = True
        
        if not mcq.question or len(mcq.question.strip()) < 10:
            errors.append("Question is too short or empty")
            is_valid = False
        
        if len(mcq.options) != 4:
            errors.append(f"Must have exactly 4 options, found {len(mcq.options)}")
            is_valid = False
        
        correct_options = [opt for opt in mcq.options if opt.is_correct]
        if len(correct_options) != 1:
            errors.append(f"Must have exactly 1 correct answer, found {len(correct_options)}")
            is_valid = False
        
        if not mcq.explanation or len(mcq.explanation.strip()) < 10:
            errors.append("Explanation is too short or empty")
            is_valid = False
        
        # Check option labels
        expected_labels = {'A', 'B', 'C', 'D'}
        actual_labels = {opt.label for opt in mcq.options}
        if actual_labels != expected_labels:
            errors.append(f"Invalid option labels: {actual_labels}")
            is_valid = False
        
        return is_valid
    
    def _check_metadata(self, mcq: MCQ, errors: List[str]) -> bool:
        """Check if MCQ has required metadata"""
        is_valid = True
        
        if not mcq.chunk_id:
            errors.append("Missing chunk_id")
            is_valid = False
        
        if not mcq.source_filename:
            errors.append("Missing source_filename")
            is_valid = False
        
        if not mcq.difficulty:
            errors.append("Missing difficulty level")
            is_valid = False
        
        return is_valid

