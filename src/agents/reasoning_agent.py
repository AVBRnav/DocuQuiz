from typing import List, Dict
from openai import OpenAI
from src.agents.retriever_agent import RetrievedChunk


class ReasoningAgent:
    """
    reasoning agent that will process retrieved chunks
    and generate comprehensive responses using an LLM.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4", 
                 temperature: float = 0.7, max_tokens: int = 2000):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = "reasoning_agent"
    
    def generate_response(self, query: str, context: str, 
                         system_prompt: str = None) -> str:
        default_system = (
            "You are a helpful AI assistant with access to a knowledge base. "
            "Use the provided context to answer questions accurately. "
            "If the context doesn't contain enough information, say so."
        )
        
        messages = [
            {"role": "system", "content": system_prompt or default_system},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content
    
    def execute(self, query: str, retrieved_chunks: List[RetrievedChunk], 
                system_prompt: str = None) -> Dict:
        context_parts = []
        for chunk in retrieved_chunks:
            context_parts.append(chunk.text)
        context = "\n\n".join(context_parts)
        
        response = self.generate_response(query, context, system_prompt)
        
        return {
            'query': query,
            'response': response,
            'num_sources': len(retrieved_chunks),
            'sources': [chunk.metadata for chunk in retrieved_chunks]
        }
