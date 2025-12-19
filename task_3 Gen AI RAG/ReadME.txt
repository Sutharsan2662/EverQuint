RAG Chatbot (Wikipedia)

A user-friendly RAG chatbot built with Streamlit that retrieves relevant information from
Wikipedia or uploaded documents and generates grounded responses using FAISS-based
semantic search, HuggingFace embeddings, and a locally hosted LLM via Ollama.

Architecture:
User Question
↓
Query Preprocessing (cleaning + acronym expansion)
↓
Document Retrieval (Wikipedia / Uploaded File)
↓
Text Chunking
↓
Embeddings (HuggingFace)
↓
Vector Search (FAISS – Top K chunks)
↓
Context Injection into Prompt
↓
LLM Generation (Ollama – TinyLlama)

