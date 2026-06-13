import streamlit as st

# =====================================================
# Project Imports
# =====================================================
from src.loader import load_uploaded_pdf
from src.model import get_embeddings, get_llm
from src.vectorstore import create_vectorstore
from src.rag_chain import build_chain
from src.history_aware_retriever import rewrite_question


# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="PDF RAG Chatbot",
    layout="wide"
)

st.title("📚 PDF RAG Chatbot")


# =====================================================
# PDF Upload Section
# =====================================================
st.sidebar.header("📄 Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF",
    type="pdf"
)


# =====================================================
# Stop App Until PDF Uploaded
# =====================================================
if uploaded_file is None:

    st.info(
        "👈 Upload a PDF from the sidebar to begin."
    )

    st.stop()


# =====================================================
# Session State Initialization
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "standalone_question" not in st.session_state:
    st.session_state.standalone_question = ""

if "retrieved_docs" not in st.session_state:
    st.session_state.retrieved_docs = []

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None


# =====================================================
# If User Uploads New PDF
# Clear Old Chat History
# =====================================================
if st.session_state.current_pdf != uploaded_file.name:

    st.session_state.current_pdf = uploaded_file.name

    st.session_state.messages = []

    st.session_state.retrieved_docs = []

    st.session_state.standalone_question = ""


# =====================================================
# Load RAG Components
# =====================================================
@st.cache_resource
def load_rag(uploaded_file):

    # ---------------------------------
    # Load PDF
    # ---------------------------------
    docs = load_uploaded_pdf(
        uploaded_file
    )

    # ---------------------------------
    # Load Local Embedding Model
    # ---------------------------------
    embeddings = get_embeddings()

    # ---------------------------------
    # Create Fresh FAISS Vector Store
    # ---------------------------------
    vectorstore = create_vectorstore(
        docs,
        embeddings
    )

    # ---------------------------------
    # Retriever
    # ---------------------------------
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # ---------------------------------
    # Gemini LLM
    # ---------------------------------
    llm = get_llm()

    # ---------------------------------
    # Build RAG Chain
    # ---------------------------------
    chain = build_chain(
        retriever,
        llm
    )

    return chain, retriever


# =====================================================
# Show Processing Spinner
# =====================================================
with st.spinner(
    "Processing PDF..."
):

    chain, retriever = load_rag(
        uploaded_file
    )


# =====================================================
# Gemini used for Question Rewriting
# =====================================================
llm = get_llm()


# =====================================================
# Show Uploaded PDF Info
# =====================================================
st.sidebar.success(
    f"Loaded: {uploaded_file.name}"
)


# =====================================================
# Display Previous Messages
# =====================================================
for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )


# =====================================================
# Chat Input
# =====================================================
question = st.chat_input(
    "Ask anything from your PDF..."
)


# =====================================================
# Handle User Question
# =====================================================
if question:

    # ---------------------------------
    # Save User Message
    # ---------------------------------
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # ---------------------------------
    # Rewrite Follow-up Questions
    # Example:
    #
    # Q1:
    # What is Logistic Regression?
    #
    # Q2:
    # What are its examples?
    #
    # Rewritten:
    # What are examples of Logistic
    # Regression?
    # ---------------------------------
    standalone_question = rewrite_question(
        llm,
        question,
        st.session_state.messages
    )

    st.session_state.standalone_question = (
        standalone_question
    )

    # ---------------------------------
    # Retrieve Documents
    # Used For Debugging
    # ---------------------------------
    retrieved_docs = retriever.invoke(
        standalone_question
    )

    st.session_state.retrieved_docs = (
        retrieved_docs
    )

    # ---------------------------------
    # Run RAG Chain
    # ---------------------------------
    answer = chain.invoke(
        {
            "question":
                standalone_question,

            "history":
                st.session_state.messages
        }
    )

    # ---------------------------------
    # Show Assistant Answer
    # ---------------------------------
    with st.chat_message(
        "assistant"
    ):
        st.markdown(answer)

    # ---------------------------------
    # Save Assistant Message
    # ---------------------------------
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# =====================================================
# SIDEBAR DEBUG PANEL
# =====================================================
st.sidebar.divider()

st.sidebar.header(
    "🔍 Retrieval Debugger"
)

# ---------------------------------
# Rewritten Question
# ---------------------------------
st.sidebar.subheader(
    "Rewritten Question"
)

st.sidebar.info(
    st.session_state.standalone_question
)

# ---------------------------------
# Retrieved Chunks
# ---------------------------------
st.sidebar.divider()

st.sidebar.subheader(
    "Retrieved Chunks"
)

if st.session_state.retrieved_docs:

    for idx, doc in enumerate(
        st.session_state.retrieved_docs,
        start=1
    ):

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        st.sidebar.markdown(
            f"### Chunk {idx}"
        )

        st.sidebar.write(
            f"📁 File: {source}"
        )

        st.sidebar.write(
            f"📄 Page: {page}"
        )

        st.sidebar.text_area(
            label=f"Chunk {idx}",
            value=doc.page_content[:500],
            height=150,
            key=f"chunk_{idx}"
        )