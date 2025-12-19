import streamlit as st
import requests
import re
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader # New Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# --- 1. CONFIGURATION ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "tinyllama"  # Memory-safe model
LOAD_MAX_DOCS = 1 
CLEAN_QUERY_DELIMITER = "---CLEAN_QUERY---"

# --- 2. CACHED RESOURCES ---

@st.cache_resource
def get_embedding_model():
    """Loads and caches the heavy embedding model once."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

# --- 3. HELPER FUNCTIONS ---

def clean_wikipedia_query(query: str) -> str:
    """Removes instructional phrases to clean the query for reliable Wikipedia keyword search."""
    phrases_to_remove = [
        r'\bexplain\b', r'\bwhat is\b', r'\bsite the source\b', 
        r'\bw\.r\.t\b', r'\bwith respect to\b', r'\bregard to\b', 
        r'\band cite the source\b', r'\bhow to\b', r'\bmeans\b', r'\bmeant\b', r'\bwhat is\b'
    ]
    
    cleaned_query = query.lower()
    for phrase in phrases_to_remove:
        cleaned_query = re.sub(phrase, '', cleaned_query).strip()

    keywords = " ".join([word for word in cleaned_query.split() if len(word) > 2])
    return keywords.strip()

def ollama_generate(prompt: str):
    """Calls Ollama API."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120) 
    except requests.exceptions.Timeout:
        return "Error: Ollama API call timed out. Is the model loaded and server running?"
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama API. Ensure Ollama is running."

    if response.status_code == 200:
        return response.json().get("response", "No response content found.")
    else:
        return f"Ollama Server Error {response.status_code}: {response.text}"

@st.cache_data(show_spinner="Expanding acronyms in query...")
def preprocess_query(raw_query: str):
    """
    Preprocessing focuses ONLY on acronym expansion for stability.
    """
    preprocessing_prompt = f"""
    Task: Expand common acronyms in the user's input query (e.g., convert 'AI' to 'Artificial Intelligence', 'ML' to 'Machine Learning').
    Output the final, expanded query on a single line, enclosed within the delimiter '{CLEAN_QUERY_DELIMITER}'. Do not add any other words, explanations, or instructions.

    Raw Query: "{raw_query}"

    {CLEAN_QUERY_DELIMITER}
    """
    
    llm_output = ollama_generate(preprocessing_prompt)
    
    if llm_output.startswith("Error"):
        return raw_query 

    try:
        start_index = llm_output.find(CLEAN_QUERY_DELIMITER)
        
        if start_index != -1:
            cleaned_query = llm_output[start_index + len(CLEAN_QUERY_DELIMITER):].strip()
            lines = cleaned_query.split('\n')
            final_query = lines[0].strip().replace('"', '')

            if not final_query:
                return raw_query
            
            return final_query
        else:
            return raw_query

    except Exception:
        return raw_query


# --- RAG FUNCTION 1: WIKIPEDIA ---

@st.cache_data(show_spinner="Searching Wikipedia and generating summary...")
def run_dynamic_rag(query: str, _embeddings): 
    """
    Caches the Wikipedia RAG result based on the user's query.
    """
    
    search_query = clean_wikipedia_query(query)
    
    if not search_query:
        search_query = query 

    try:
        loader = WikipediaLoader(query=search_query, load_max_docs=LOAD_MAX_DOCS)
        all_pages = loader.load()
    except Exception:
        return "Error connecting to Wikipedia. Please check your network connection.", None, None

    if not all_pages:
        return f"Wikipedia search returned no relevant results for '{search_query}'.", None, None

    source_title = all_pages[0].metadata.get("title", "Unknown Source")
    source_url = all_pages[0].metadata.get("source", "#")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_pages)
    
    vector_db = FAISS.from_documents(chunks, _embeddings) 
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(query) 
    
    context = "\n---\n".join([doc.page_content for doc in retrieved_docs])

    # 4. Build Prompt (UPDATED: Priority for Definition/Explanation)
    prompt = f"""
    You are an expert assistant. Your goal is to provide a clear explanation that is easily understandable to a layman but is technically accurate enough for an expert.
    
    Summarize the following retrieved context to answer the user's question.
    Ensure your answer is based ONLY on the provided context.

    User Question: {query}

    Context retrieved from Wikipedia:
    ---
    {context}
    ---
    
    Summary should be structured into at least three distinct paragraphs. The **FIRST PARAGRAPH MUST BE DEDICATED TO A CLEAR, CONCISE DEFINITION AND EXPLANATION** of what the subject of the question is. The subsequent paragraphs can cover features, mechanics, or related concepts.
    """

    ai_summary = ollama_generate(prompt)
    
    return ai_summary, source_title, source_url


# --- RAG FUNCTION 2: DOCUMENT PROCESSING (NEW) ---

@st.cache_data(show_spinner="Processing document and generating summary...")
def run_document_rag(uploaded_file, query: str, _embeddings):
    """Handles document RAG, saving file, loading, splitting, and summarization."""
    # 1. Save uploaded file temporarily
    temp_file_path = os.path.join("./temp_docs", uploaded_file.name)
    os.makedirs("./temp_docs", exist_ok=True)
    
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 2. Load the document based on its extension
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    if file_extension == "pdf":
        loader = PyPDFLoader(temp_file_path)
    elif file_extension == "docx":
        loader = Docx2txtLoader(temp_file_path)
    elif file_extension == "txt":
        loader = TextLoader(temp_file_path)
    else:
        os.remove(temp_file_path)
        return "Unsupported file type. Please upload a PDF, DOCX, or TXT file.", None, None

    # 3. Load, Split, and Index
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    vector_db = FAISS.from_documents(chunks, _embeddings) 
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(query) 
    
    context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
    
    # 4. Build Prompt (UPDATED: Priority for Definition/Explanation)
    prompt = f"""
    You are an expert assistant. Your goal is to provide a clear explanation that is easily understandable to a layman but is technically accurate enough for an expert.
    
    Summarize the following retrieved context to answer the user's question.
    Ensure your answer is based ONLY on the provided context.

    User Question: {query}

    Context retrieved from Document ({uploaded_file.name}):
    ---
    {context}
    ---
    
    Summary should be structured into at least three distinct paragraphs. The **FIRST PARAGRAPH MUST BE DEDICATED TO A CLEAR, CONCISE DEFINITION AND EXPLANATION** of what the subject of the question is. The subsequent paragraphs can cover features, mechanics, or related concepts.
    """
    
    # 5. Generate Response
    ai_summary = ollama_generate(prompt)

    # 6. Clean up the temporary file
    try:
        os.remove(temp_file_path)
    except OSError as e:
        st.warning(f"Error cleaning up temp file: {e}")
    
    source_title = uploaded_file.name
    
    return ai_summary, source_title, "" # Empty URL for local files

# --- 4. STREAMLIT APP LAYOUT ---

def main():
    st.set_page_config(page_title="RAG Chatbot", layout="centered")
    st.title("🧠 Simple RAG Chatbot") 
    st.caption(f"Powered by **{OLLAMA_MODEL}** | Source: Wikipedia / Uploaded Documents")
    st.markdown("---")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    embeddings = get_embedding_model()
    
    # 4a. Document Uploader in Sidebar (NEW)
    st.sidebar.header("Document RAG")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF, DOCX, or TXT file for context:",
        type=["pdf", "docx", "txt"]
    )
    st.sidebar.caption("If a file is uploaded, the chat will use the document for answers.")


    # 4b. Display all previous messages
    for message in st.session_state.messages:
        
        # 1. User Question 
        st.markdown(f"**Question:** *{message['original_question']}*")
        
        if message['original_question'] != message['processed_question']:
            st.caption(f"*(Query expanded to: {message['processed_question']})*")
        
        # 2. Answer (Minimalist Display: Just the summary)
        st.markdown(message['summary'])
        
        # Check for contextually relevant diagrams
        processed_q = message['processed_question'].lower()
        
        if "camera" in processed_q or "how do cameras work" in processed_q:
            st.markdown("")

        if "artificial intelligence" in processed_q or "machine learning" in processed_q or "neural network" in processed_q:
            st.markdown("")
        
        # 3. Source Link
        st.markdown(f"---")
        # Display Source based on type (Wikipedia or Document)
        if message.get('source_type') == "Document":
             st.markdown(f"**Source:** Uploaded Document ({message['title']})") 
        else:
             st.markdown(f"**Source:** Wikipedia [{message['title']}]({message['url']})") 
             
        st.markdown("<br>", unsafe_allow_html=True) 

    # 4c. Get user input
    user_query = st.chat_input(
        "Ask a question on any topic or about the uploaded document..."
    )

    if user_query:
        # 1. PRE-PROCESS THE QUERY
        processed_query = preprocess_query(user_query)
        
        # 2. Run the RAG process: Document RAG if file is uploaded, otherwise Wikipedia RAG
        if uploaded_file is not None:
            ai_summary, source_title, source_url = run_document_rag(uploaded_file, processed_query, embeddings)
            source_type = "Document"
        else:
            ai_summary, source_title, source_url = run_dynamic_rag(processed_query, embeddings)
            source_type = "Wikipedia"

        # 3. Store new turn and re-run
        if source_title and not ai_summary.startswith("Error"):
            st.session_state.messages.append({
                "original_question": user_query,
                "processed_question": processed_query,
                "summary": ai_summary,
                "title": source_title,
                "url": source_url,
                "source_type": source_type # Stored to differentiate source display
            })
            st.rerun() 
        else:
            st.warning(ai_summary)

if __name__ == "__main__":
    main()