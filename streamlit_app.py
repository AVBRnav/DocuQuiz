#!/usr/bin/env python3
"""
DocuQuiz - Streamlit Web Interface
Beautiful and minimal UI for MCQ generation
"""

import streamlit as st
import json
from datetime import datetime
from src.config_loader import ConfigLoader
from agent_pipeline import AgenticMCQPipeline

# Page configuration
st.set_page_config(
    page_title="DocuQuiz - MCQ Generator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .mcq-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 15px auto;
        border-left: 4px solid #1f77b4;
        max-width: 900px;
    }
    .option-correct {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        color: #155724;
        max-width: 100%;
    }
    .option-incorrect {
        background-color: #f8f9fa;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        color: #262730;
        max-width: 100%;
    }
    .quality-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
    }
    .badge-high {
        background-color: #d4edda;
        color: #155724;
    }
    .badge-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    .badge-low {
        background-color: #f8d7da;
        color: #721c24;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        padding: 0.6rem;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #155a8a;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_pipeline():
    """Initialize the MCQ pipeline (cached for performance)"""
    try:
        config = ConfigLoader()
        pipeline = AgenticMCQPipeline(config)
        return pipeline, None
    except Exception as e:
        return None, str(e)


def display_mcq(mcq, index, critique=None):
    """Display a single MCQ with beautiful formatting"""
    
    with st.container():
        st.markdown(f"<div class='mcq-container'>", unsafe_allow_html=True)
        
        # Question header
        st.markdown(f"### 📝 Question {index}")
        st.markdown(f"**{mcq['question']}**")
        
        st.markdown("---")
        
        # Options
        st.markdown("**Options:**")
        for opt in mcq['options']:
            if opt['is_correct']:
                st.markdown(f"""
                <div class='option-correct'>
                    ✅ <strong>{opt['label']}.</strong> {opt['text']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='option-incorrect'>
                    ⚪ <strong>{opt['label']}.</strong> {opt['text']}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Answer and explanation
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"**✓ Correct Answer:** `{mcq['correct_answer']}`")
        with col2:
            st.markdown(f"**🎯 Difficulty:** `{mcq['difficulty']}`")
        
        with st.expander("📖 Explanation"):
            st.write(mcq['explanation'])
        
        # Metadata
        with st.expander("ℹ️ Metadata & Quality"):
            meta_col1, meta_col2 = st.columns(2)
            with meta_col1:
                st.markdown(f"**Source:** {mcq['source_filename']}")
                st.markdown(f"**Chunk ID:** {mcq['chunk_id']}")
            
            if critique:
                with meta_col2:
                    st.markdown("**Quality Scores:**")
                    st.markdown(f"- Clarity: {critique['clarity_score']:.1f}/10")
                    st.markdown(f"- Correctness: {critique['correctness_score']:.1f}/10")
                    st.markdown(f"- Grounding: {critique['grounding_score']:.1f}/10")
                    st.markdown(f"- **Overall: {critique['overall_score']:.1f}/10**")
        
        st.markdown("</div>", unsafe_allow_html=True)


def get_quality_badge(score):
    """Get quality badge HTML based on score"""
    if score >= 8.0:
        return f"<span class='quality-badge badge-high'>Excellent ({score:.1f}/10)</span>"
    elif score >= 6.0:
        return f"<span class='quality-badge badge-medium'>Good ({score:.1f}/10)</span>"
    else:
        return f"<span class='quality-badge badge-low'>Needs Review ({score:.1f}/10)</span>"


def main():
    # Header
    st.markdown("<h1 class='main-header'>🎓 DocuQuiz</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>AI-Powered MCQ Generation from Your Documents</p>", unsafe_allow_html=True)
    
    # Initialize pipeline
    with st.spinner("🔧 Initializing MCQ Generation Pipeline..."):
        pipeline, error = initialize_pipeline()
    
    if error:
        st.error(f"❌ Failed to initialize pipeline: {error}")
        st.info("💡 Make sure you have:\n1. Set up `.env` with API keys\n2. Run `python ingest_documents.py` first")
        return
    
    # Sidebar - Configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Number of MCQs
        num_mcqs = st.slider(
            "Number of MCQs",
            min_value=1,
            max_value=10,
            value=3,
            help="How many questions to generate"
        )
        
        # Difficulty level
        difficulty = st.selectbox(
            "Difficulty Level",
            options=["Any", "easy", "medium", "hard"],
            index=0,
            help="Select difficulty level (Any = mixed)"
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            show_invalid = st.checkbox(
                "Show Invalid MCQs",
                value=False,
                help="Display MCQs that didn't pass validation"
            )
            
            top_k = st.slider(
                "Context Chunks (top_k)",
                min_value=3,
                max_value=10,
                value=5,
                help="Number of document chunks to retrieve"
            )
        
        st.markdown("---")
        
        # Info section
        st.markdown("### 📚 About")
        st.info(
            "DocuQuiz uses multi-agent AI to generate "
            "high-quality MCQs from your documents. "
            "Each question is context-grounded and validated."
        )
        
        # Quick stats
        with st.expander("📊 Quick Stats"):
            if 'last_result' in st.session_state:
                result = st.session_state.last_result
                st.metric("Total Generated", result['total_mcqs'])
                st.metric("Valid MCQs", result['valid_count'])
                st.metric("Invalid MCQs", result['invalid_count'])
    
    # Main content area
    st.markdown("## 💬 Enter Your Query")
    
    # Query input
    query = st.text_area(
        "What topic would you like MCQs about?",
        placeholder="e.g., What is the attention mechanism in transformers?",
        height=100,
        help="Enter a specific topic or question related to your documents"
    )
    
    # Example queries
    with st.expander("💡 Example Queries"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - What is the transformer architecture?
            - Explain self-attention mechanism
            - How does multi-head attention work?
            """)
        with col2:
            st.markdown("""
            - What are encoder and decoder in transformers?
            - Explain positional encoding
            - What is the purpose of attention mechanism?
            """)
    
    # Generate button
    st.markdown("")
    generate_col1, generate_col2, generate_col3 = st.columns([1, 2, 1])
    with generate_col2:
        generate_button = st.button("🚀 Generate MCQs", use_container_width=True)
    
    # Generate MCQs
    if generate_button:
        if not query.strip():
            st.warning("⚠️ Please enter a query first!")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Retrieve
            status_text.text("🔍 Retrieving relevant context...")
            progress_bar.progress(25)
            
            # Step 2: Generate
            status_text.text("✍️ Generating MCQs...")
            progress_bar.progress(50)
            
            # Generate MCQs
            difficulty_param = None if difficulty == "Any" else difficulty
            result = pipeline.generate_mcqs(
                query=query,
                num_mcqs=num_mcqs,
                difficulty=difficulty_param
            )
            
            # Step 3: Critique
            status_text.text("🎯 Evaluating quality...")
            progress_bar.progress(75)
            
            # Step 4: Complete
            status_text.text("✅ Complete!")
            progress_bar.progress(100)
            
            # Store result in session state
            st.session_state.last_result = result
            
            # Clear progress indicators
            import time
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.markdown("---")
            st.markdown("## 📋 Generated MCQs")
            
            # Summary metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("Total Generated", result['total_mcqs'])
            with metric_col2:
                st.metric("Valid MCQs", result['valid_count'], delta=f"{result['valid_count']/result['total_mcqs']*100:.0f}%")
            with metric_col3:
                avg_score = sum(c['overall_score'] for c in result['critiques']) / len(result['critiques']) if result['critiques'] else 0
                st.metric("Avg Quality", f"{avg_score:.1f}/10")
            with metric_col4:
                st.metric("Invalid", result['invalid_count'])
            
            # Display valid MCQs
            if result['valid_count'] > 0:
                st.markdown("### ✅ Valid MCQs")
                for i, mcq in enumerate(result['valid_mcqs'], 1):
                    # Find corresponding critique
                    mcq_idx = result['mcqs'].index(next(m for m in result['mcqs'] if m['question'] == mcq['question']))
                    critique = result['critiques'][mcq_idx] if mcq_idx < len(result['critiques']) else None
                    display_mcq(mcq, i, critique)
            else:
                st.warning("⚠️ No valid MCQs generated. Try adjusting your query or settings.")
            
            # Display invalid MCQs if requested
            if show_invalid and result['invalid_count'] > 0:
                st.markdown("### ⚠️ Invalid MCQs (For Review)")
                st.caption("These MCQs didn't pass validation but may still be useful with manual review.")
                for i, mcq in enumerate(result['invalid_mcqs'], 1):
                    mcq_idx = result['mcqs'].index(next(m for m in result['mcqs'] if m['question'] == mcq['question']))
                    critique = result['critiques'][mcq_idx] if mcq_idx < len(result['critiques']) else None
                    validation = result['validations'][mcq_idx] if mcq_idx < len(result['validations']) else None
                    
                    with st.expander(f"Invalid MCQ {i}: {mcq['question'][:60]}..."):
                        display_mcq(mcq, i, critique)
                        if validation and validation['validation_errors']:
                            st.error("**Validation Errors:**")
                            for error in validation['validation_errors']:
                                st.write(f"- {error}")
            
            # Download options
            st.markdown("---")
            st.markdown("### 💾 Download Results")
            
            download_col1, download_col2 = st.columns(2)
            
            with download_col1:
                # JSON download
                json_data = json.dumps(result, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_data,
                    file_name=f"mcqs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with download_col2:
                # Text download
                text_data = generate_text_export(result)
                st.download_button(
                    label="📄 Download as Text",
                    data=text_data,
                    file_name=f"mcqs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
        except Exception as e:
            st.error(f"❌ Error generating MCQs: {str(e)}")
            with st.expander("🔍 Error Details"):
                st.exception(e)


def generate_text_export(result):
    """Generate plain text export of MCQs"""
    lines = []
    lines.append("=" * 70)
    lines.append("DOCUQUIZ - GENERATED MCQs")
    lines.append("=" * 70)
    lines.append(f"\nQuery: {result['query']}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total MCQs: {result['total_mcqs']}")
    lines.append(f"Valid MCQs: {result['valid_count']}")
    lines.append(f"Invalid MCQs: {result['invalid_count']}")
    lines.append("\n" + "=" * 70)
    
    for i, mcq in enumerate(result['valid_mcqs'], 1):
        lines.append(f"\n\n{'─' * 70}")
        lines.append(f"MCQ #{i}")
        lines.append(f"{'─' * 70}")
        lines.append(f"\nQuestion: {mcq['question']}")
        lines.append(f"\nOptions:")
        for opt in mcq['options']:
            marker = "✓" if opt['is_correct'] else " "
            lines.append(f"  [{marker}] {opt['label']}. {opt['text']}")
        lines.append(f"\nCorrect Answer: {mcq['correct_answer']}")
        lines.append(f"\nExplanation: {mcq['explanation']}")
        lines.append(f"\nDifficulty: {mcq['difficulty']}")
        lines.append(f"Source: {mcq['source_filename']}")
        lines.append(f"Chunk ID: {mcq['chunk_id']}")
    
    lines.append("\n\n" + "=" * 70)
    lines.append("END OF MCQs")
    lines.append("=" * 70)
    
    return "\n".join(lines)


if __name__ == "__main__":
    main()

