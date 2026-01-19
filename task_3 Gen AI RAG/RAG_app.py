import streamlit as st
import requests
import re
import os
from langchain_community.document_loaders import (
    WikipediaLoader, PyPDFLoader, 
    Docx2txtLoader, TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- SYSTEM CONFIGURATION ---
OLLAMA_API = "http://localhost:11434/api/generate"
AI_MODEL = "mistral:7b-instruct-v0.2-q2_K"
TEMP_DIR = "./workspace_docs"

@st.cache_resource
def initialize_embeddings():
    """ Load the lightweight sentence-transformer model. """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- DOCUMENT UTILITIES ---

def build_search_index(uploaded_file, embedding_engine):
    """ Reads and prepares the document for semantic search. """
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        
    path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    ext = uploaded_file.name.split(".")[-1].lower()
    if ext == "pdf":
        loader = PyPDFLoader(path)
    elif ext == "docx":
        loader = Docx2txtLoader(path)
    else:
        loader = TextLoader(path)

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    
    # Generate and store embeddings
    vector_store = FAISS.from_documents(chunks, embedding_engine)
    os.remove(path)
    return vector_store

def clean_search_term(text):
    text = text.lower()
    ignore_phrases = [r'\bexplain\b', r'\bwhat is\b', r'\bwho is\b', r'\btell me about\b']
    for phrase in ignore_phrases:
        text = re.sub(phrase, '', text).strip()
    return text

# --- LLM ENGINE ---

def call_ollama(final_prompt):
    """ Patient wrapper for local model inference. """
    payload = {
        "model": AI_MODEL, 
        "prompt": final_prompt, 
        "stream": False,
        "keep_alive": -1 
    }
    try:
        # High timeout to prevent Read timed out errors
        response = requests.post(OLLAMA_API, json=payload, timeout=300)
        return response.json().get("response", "No output generated.")
    except Exception as e:
        return f"Connection error: {e}"

def generate_rag_response(query, db):
    """ Retrieves context and crafts the answer. """
    relevant_docs = db.similarity_search(query, k=2)
    context_text = "\n---\n".join([d.page_content for d in relevant_docs])

    prompt = f"""<s>[INST] You are a professional researcher. Use the context to answer.
    Context: {context_text}
    Question: {query}
    Instructions:
    - For facts, be concise.
    - For complex topics, use exactly two paragraphs.
    - Do NOT use labels like 'Para 1'.
    [/INST] </s>"""
    
    return call_ollama(prompt)

# --- UI LAYER ---

def main():
    st.set_page_config(page_title="Pro RAG Assistant", layout="centered")
    st.title("🚀 Smart RAG Assistant")

    # Session State Initialization
    if "doc_history" not in st.session_state: st.session_state.doc_history = []
    if "wiki_history" not in st.session_state: st.session_state.wiki_history = []
    if "doc_index" not in st.session_state: st.session_state.doc_index = None
    if "active_doc_name" not in st.session_state: st.session_state.active_doc_name = None

    embed_tool = initialize_embeddings()

    # Sidebar: File Upload and Mode Toggle
    with st.sidebar:
        st.header("📁 Knowledge Upload")
        file = st.file_uploader("Upload local context:", type=["pdf", "docx", "txt"])
        
        if file and st.session_state.active_doc_name != file.name:
            with st.status("Building Index...") as status:
                st.session_state.doc_index = build_search_index(file, embed_tool)
                st.session_state.active_doc_name = file.name
                status.update(label="Document Ready!", state="complete")
        
        st.divider()
        st.header("⚙️ Settings")
        # SEPARATE BUTTONS (Mode Selector)
        mode = st.radio("Select Research Mode:", ["📄 Document Chat", "🌐 Wikipedia Chat"])

    # --- MAIN CONTENT AREA (NO BOXES) ---
    if mode == "📄 Document Chat":
        st.subheader(f"Chatting with: {st.session_state.active_doc_name or 'No File'}")
        
        # Standard flow without container
        for msg in st.session_state.doc_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        doc_query = st.chat_input("Ask about your document...")
        if doc_query:
            st.session_state.doc_history.append({"role": "user", "content": doc_query})
            st.rerun() # Forces display of user message immediately
            
    elif mode == "🌐 Wikipedia Chat":
        st.subheader("General Knowledge (Wikipedia)")
        
        for msg in st.session_state.wiki_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        wiki_query = st.chat_input("Ask a general question...")
        if wiki_query:
            st.session_state.wiki_history.append({"role": "user", "content": wiki_query})
            st.rerun()

    # Logic to handle processing after the rerun/user input
    # Document Logic
    if st.session_state.doc_history and st.session_state.doc_history[-1]["role"] == "user" and mode == "📄 Document Chat":
        if st.session_state.doc_index:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing PDF..."):
                    ans = generate_rag_response(st.session_state.doc_history[-1]["content"], st.session_state.doc_index)
                    st.markdown(ans)
                    st.session_state.doc_history.append({"role": "assistant", "content": ans})
        else:
            st.warning("Please upload a file first.")

    # Wikipedia Logic
    if st.session_state.wiki_history and st.session_state.wiki_history[-1]["role"] == "user" and mode == "🌐 Wikipedia Chat":
        with st.chat_message("assistant"):
            with st.spinner("Searching Wikipedia..."):
                query = st.session_state.wiki_history[-1]["content"]
                keyword = clean_search_term(query)
                loader = WikipediaLoader(query=keyword, load_max_docs=1)
                wiki_results = loader.load()
                
                if wiki_results:
                    temp_wiki_db = FAISS.from_documents(wiki_results, embed_tool)
                    ans = generate_rag_response(query, temp_wiki_db)
                    st.markdown(ans)
                    st.session_state.wiki_history.append({"role": "assistant", "content": ans})
                else:
                    st.error("No Wikipedia page found.")

if __name__ == "__main__":
    main()
