"""
MCQ Data Models and Schemas
Defines structured data models for MCQ generation workflow.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class DifficultyLevel(Enum):
    """MCQ difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ValidationStatus(Enum):
    """Validation status for MCQs"""
    VALID = "valid"
    INVALID = "invalid"
    NEEDS_REVISION = "needs_revision"


@dataclass
class MCQOption:
    """Represents a single MCQ option"""
    label: str  # A, B, C, D
    text: str
    is_correct: bool = False


@dataclass
class MCQ:
    """Represents a Multiple Choice Question"""
    question: str
    options: List[MCQOption]
    correct_answer: str  # The label of correct option (A, B, C, or D)
    explanation: str
    difficulty: DifficultyLevel
    chunk_id: str
    source_filename: str
    context_snippet: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert MCQ to dictionary format"""
        return {
            "question": self.question,
            "options": [
                {
                    "label": opt.label,
                    "text": opt.text,
                    "is_correct": opt.is_correct
                }
                for opt in self.options
            ],
            "correct_answer": self.correct_answer,
            "explanation": self.explanation,
            "difficulty": self.difficulty.value,
            "chunk_id": self.chunk_id,
            "source_filename": self.source_filename,
            "context_snippet": self.context_snippet,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCQ':
        """Create MCQ from dictionary"""
        options = [
            MCQOption(
                label=opt["label"],
                text=opt["text"],
                is_correct=opt.get("is_correct", False)
            )
            for opt in data["options"]
        ]
        
        return cls(
            question=data["question"],
            options=options,
            correct_answer=data["correct_answer"],
            explanation=data["explanation"],
            difficulty=DifficultyLevel(data["difficulty"]),
            chunk_id=data["chunk_id"],
            source_filename=data["source_filename"],
            context_snippet=data.get("context_snippet", ""),
            metadata=data.get("metadata", {})
        )


@dataclass
class CritiqueResult:
    """Represents the critique of an MCQ"""
    mcq_index: int
    clarity_score: float  # 0-10
    correctness_score: float  # 0-10
    grounding_score: float  # 0-10
    difficulty_assessment: DifficultyLevel
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    overall_score: float = 0.0
    
    def __post_init__(self):
        """Calculate overall score"""
        self.overall_score = (
            self.clarity_score + 
            self.correctness_score + 
            self.grounding_score
        ) / 3.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert critique to dictionary"""
        return {
            "mcq_index": self.mcq_index,
            "clarity_score": self.clarity_score,
            "correctness_score": self.correctness_score,
            "grounding_score": self.grounding_score,
            "overall_score": self.overall_score,
            "difficulty_assessment": self.difficulty_assessment.value,
            "issues": self.issues,
            "suggestions": self.suggestions
        }


@dataclass
class ValidationResult:
    """Represents validation result for an MCQ"""
    mcq_index: int
    status: ValidationStatus
    is_context_grounded: bool
    is_properly_formatted: bool
    has_required_metadata: bool
    has_hallucination: bool
    validation_errors: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if MCQ is valid"""
        return (
            self.status == ValidationStatus.VALID and
            self.is_context_grounded and
            self.is_properly_formatted and
            self.has_required_metadata and
            not self.has_hallucination
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary"""
        return {
            "mcq_index": self.mcq_index,
            "status": self.status.value,
            "is_valid": self.is_valid(),
            "is_context_grounded": self.is_context_grounded,
            "is_properly_formatted": self.is_properly_formatted,
            "has_required_metadata": self.has_required_metadata,
            "has_hallucination": self.has_hallucination,
            "validation_errors": self.validation_errors
        }


@dataclass
class MCQGenerationResult:
    """Complete result from MCQ generation workflow"""
    query: str
    mcqs: List[MCQ]
    critiques: List[CritiqueResult]
    validations: List[ValidationResult]
    valid_mcqs: List[MCQ] = field(default_factory=list)
    invalid_mcqs: List[MCQ] = field(default_factory=list)
    
    def __post_init__(self):
        """Separate valid and invalid MCQs"""
        for i, mcq in enumerate(self.mcqs):
            validation = self.validations[i] if i < len(self.validations) else None
            if validation and validation.is_valid():
                self.valid_mcqs.append(mcq)
            else:
                self.invalid_mcqs.append(mcq)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "query": self.query,
            "total_mcqs": len(self.mcqs),
            "valid_count": len(self.valid_mcqs),
            "invalid_count": len(self.invalid_mcqs),
            "mcqs": [mcq.to_dict() for mcq in self.mcqs],
            "critiques": [critique.to_dict() for critique in self.critiques],
            "validations": [validation.to_dict() for validation in self.validations],
            "valid_mcqs": [mcq.to_dict() for mcq in self.valid_mcqs]
        }

