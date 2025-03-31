from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import faiss
import numpy as np
import openai
import os
import streamlit as st

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
openai_client = openai.Client(api_key=api_key)


# Streamlit UI setup
st.set_page_config(page_title="PDF Chatbot", layout="wide")


if "index" not in st.session_state:
    st.session_state.index = None
if "texts" not in st.session_state:
    st.session_state.texts = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

def load_and_chunk_pdf(uploaded_file, chunk_size=500, chunk_overlap=100):
    """
    Loads a PDF file from Streamlit's file uploader, extracts text, and splits it into manageable chunks.
    
    Parameters:
    - uploaded_file (BytesIO): The uploaded PDF file.
    - chunk_size (int): The maximum number of characters per chunk.
    - chunk_overlap (int): Overlap between chunks to preserve context.
    
    Returns:
    - List of text chunks extracted from the PDF.
    """
    try:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        texts = [chunk.page_content for chunk in chunks]
        os.remove("temp.pdf")  # Cleanup
        return texts
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return []

def generate_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    embedding_model = SentenceTransformer(model_name)
    embeddings = embedding_model.encode(texts, batch_size=32, convert_to_tensor=True)
    print(f"Generated {len(embeddings)} embeddings")
    return embedding_model, embeddings


def create_faiss_index(embeddings):
    embeddings_np = np.array([embedding.cpu().numpy() for embedding in embeddings])
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    print(f"FAISS index created with {index.ntotal} vectors")
    return index

def search_faiss(index, query, texts, embedding_model, top_k=3):
    query = query.strip().lower()
    print(f"Searching FAISS index for query: '{query}'")

    query_embedding = (
        embedding_model.encode([query], convert_to_tensor=True).cpu().numpy()
    )
    distances, indices = index.search(query_embedding, top_k)
    results = [texts[i] for i in indices[0] if i < len(texts)]

    return results if results else ["No relevant results found."]

def generate_response(query, retrieved_chunks):
    if not retrieved_chunks:
        return "No relevant information found in the document."

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant."},
                {
                    "role": "user",
                    "content": f"Here are relevant document excerpts:\n{chr(10).join(retrieved_chunks)}",
                },
                {"role": "user", "content": query},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"



# Sidebar for file upload
with st.sidebar:
    st.header("Upload a PDF")
    
    # Keep file uploader visible even after uploading
    uploaded_file = st.file_uploader("", type=["pdf"], key="file_uploader")

    if uploaded_file is not None and "uploaded_file" not in st.session_state:
        with st.spinner("Processing PDF..."):
            texts = load_and_chunk_pdf(uploaded_file)
            embedding_model, embeddings = generate_embeddings(texts)
            index = create_faiss_index(embeddings)
            
            st.session_state.texts = texts
            st.session_state.index = index
            st.session_state.embedding_model = embedding_model
            st.session_state.uploaded_file = uploaded_file.name

        st.markdown("<p style='color:green;'>File is ready to process</p>", unsafe_allow_html=True)

    # Show uploaded file name & option to remove it


st.title("💬 Chat with your PDF")
st.caption("🚀 A Rag implementation Demo")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask any question from the file uploaded?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    if st.session_state.index is not None:
        retrieved_chunks = search_faiss(
            st.session_state.index, prompt, st.session_state.texts, st.session_state.embedding_model
        )
        if retrieved_chunks is None:
            retrieved_chunks="No data retreived"
        st.session_state.messages.append({"role": "assistant", "content": retrieved_chunks})
        st.chat_message("assistant").write(retrieved_chunks)
        response = generate_response(prompt, retrieved_chunks)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)



