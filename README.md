# DocuQuiz
An intelligent multi-agent pipeline ensuring factual accuracy in automated MCQ generation from domain-specific documents

</div>

---

## 🌟 Overview

**DocuQuiz** is an intelligent MCQ generation system that uses **multi-agent AI architecture** to create context-grounded, validated multiple choice questions from your document corpus. Built with CrewAI, it ensures every question is accurate, relevant, and pedagogically sound.

### ✨ Key Highlights

- 🤖 **Multi-Agent System** - 4 specialized AI agents working together
- 🎯 **Context-Grounded** - Zero hallucinations, all answers derived from your documents
- 📊 **Quality Control** - Multi-layer validation (Generation → Critique → Validation)
- 🌐 **Beautiful Web UI** - Clean Streamlit interface for easy use
- 🔍 **Source Traceability** - Every MCQ linked to source document and chunk
- 📈 **Quality Metrics** - Clarity, Correctness, and Grounding scores (0-10)

---

## 🎬 Demo
### Web Interface
https://github.com/user-attachments/assets/479e1273-242a-42d0-bd81-dc880e640581



### Generated MCQ Example
```
Question: What is the primary innovation of the Transformer architecture?

Options:
  ✅ A. Self-attention mechanism
  ⚪ B. Recurrent connections
  ⚪ C. Convolutional layers
  ⚪ D. Pooling operations

Correct Answer: A
Explanation: The Transformer introduced the self-attention mechanism, 
allowing it to model dependencies without recurrence.

Quality Scores: Clarity: 9.0/10 | Correctness: 9.5/10 | Grounding: 9.0/10
Source: transformer_paper.pdf (Chunk #5)
```

---

## ✨ Features

### 🤖 Multi-Agent Architecture

```mermaid
graph LR
    A[User Query] --> B[Retrieval Agent]
    B --> C[Generation Agent]
    C --> D[Critic Agent]
    D --> E[Validation Agent]
    E --> F[Valid MCQs]
```

**4 Specialized Agents:**
1. **Retrieval Agent** - Fetches relevant document chunks from vector database
2. **Generation Agent** - Creates MCQs using GPT-4, strictly from context
3. **Critic Agent** - Evaluates quality on 3 dimensions (Clarity, Correctness, Grounding)
4. **Validation Agent** - Enforces formatting and quality standards

### 🎨 Beautiful Web Interface

- **Intuitive UI** - Clean, minimal Streamlit interface
- **Real-time Progress** - Watch MCQs being generated step-by-step
- **Interactive Config** - Adjust settings (difficulty, count) on the fly
- **Quality Visualization** - Color-coded badges and expandable details
- **Export Options** - Download as JSON or formatted text

### 📊 Quality Assurance

- **3-Dimensional Scoring** - Clarity, Correctness, Grounding (0-10 each)
- **Format Validation** - Ensures 4 options, 1 correct answer, proper structure
- **Hallucination Detection** - Verifies all content is grounded in source
- **Metadata Tracking** - Full traceability to source document and chunk

### 🔧 Flexible Configuration

- **Difficulty Levels** - Easy, Medium, Hard, or Mixed
- **Batch Processing** - Generate for multiple topics at once
- **Adjustable Context** - Control number of retrieved chunks (top-k)
- **Custom Thresholds** - Fine-tune quality and relevance scores

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/docuquiz.git
cd docuquiz

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your_openai_key
# PINECONE_API_KEY=your_pinecone_key
# PINECONE_ENVIRONMENT=your_environment
```

### Usage

#### 1. Ingest Your Documents

```bash
# Add your documents to the docs/ folder
cp your_documents.pdf docs/

# Run ingestion
python ingest_documents.py
```

#### 2. Launch Web Interface (Recommended)

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

#### 3. Or Use Command Line

```bash
# Interactive mode
python agent_pipeline.py

# Direct query
python agent_pipeline.py "What is the attention mechanism?"
```

---

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────┐
│         Web Interface (Streamlit)       │
├─────────────────────────────────────────┤
│        MCQ Orchestrator (Coordinator)   │
├─────────────────────────────────────────┤
│  Retrieval │ Generation │ Critic │ Val │
├─────────────────────────────────────────┤
│    Embeddings    │    Vector Store      │
│    (OpenAI)      │    (Pinecone)        │
├─────────────────────────────────────────┤
│         Document Processing             │
│   (Loading, Chunking, Embedding)        │
└─────────────────────────────────────────┘
```

### Tech Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-4
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector DB**: Pinecone
- **Agent Framework**: CrewAI
- **Language**: Python 3.8+

### Data Flow

1. **Document Ingestion**
   - Load PDFs/text files → Chunk into smaller pieces → Generate embeddings → Store in Pinecone

2. **MCQ Generation**
   - User query → Embed query → Retrieve similar chunks → Generate MCQs → Critique → Validate → Return

3. **Quality Control**
   - Each MCQ scored on Clarity, Correctness, Grounding
   - Format validation (4 options, 1 correct, proper structure)
   - Source verification (no hallucinations)

---


---

## 📊 Example Output

### JSON Format
```json
{
  "query": "What is attention mechanism?",
  "total_mcqs": 5,
  "valid_count": 4,
  "valid_mcqs": [
    {
      "question": "What is the primary purpose of attention?",
      "options": [
        {"label": "A", "text": "To focus on relevant parts", "is_correct": true},
        {"label": "B", "text": "To increase model size", "is_correct": false},
        {"label": "C", "text": "To reduce computation", "is_correct": false},
        {"label": "D", "text": "To normalize weights", "is_correct": false}
      ],
      "correct_answer": "A",
      "explanation": "According to the context...",
      "difficulty": "medium",
      "chunk_id": "5",
      "source_filename": "paper.pdf",
      "quality_scores": {
        "clarity": 9.0,
        "correctness": 9.5,
        "grounding": 9.0,
        "overall": 9.2
      }
    }
  ]
}
```

---








## 📝 Project Structure

```
docuquiz/
├── streamlit_app.py          # Web interface
├── agent_pipeline.py         # CLI interface
├── ingest_documents.py       # Document ingestion
├── src/
│   ├── mcq_models.py         # Data models
│   ├── agents/
│   │   ├── mcq_agents.py     # AI agents
│   │   └── mcq_orchestrator.py
│   └── [core modules]
├── config/
│   └── config.yaml           # Configuration
├── docs/                     # Your documents
├── examples/                 # Usage examples
└── requirements.txt          # Dependencies
```

---

<div align="center">

</div>

