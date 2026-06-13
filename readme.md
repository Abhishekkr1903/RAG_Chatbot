# 📚 PDF RAG Chatbot using Gemini & LangChain

A production-style Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and interact with them through a conversational AI chatbot.

The system uses LangChain, FAISS, HuggingFace Embeddings, and Google Gemini to retrieve relevant information from PDFs and generate context-aware answers.

---

# 🚀 Features

## Core RAG Features

✅ Upload PDF directly from the UI

✅ Semantic Search using FAISS

✅ Local Embeddings using Sentence Transformers

✅ Gemini 2.5 Flash for Answer Generation

✅ Source Page Tracking

✅ Context-Aware Question Answering

---

## Conversational AI Features

✅ Multi-turn Conversations

✅ Chat History Management

✅ History-Aware Retrieval

✅ Follow-up Question Understanding

### Example

**User:**

> What is Linear Regression?

**User:**

> What are its examples?

The system automatically rewrites the question to:

> What are examples of Linear Regression?

before retrieval.

---

## Debugging Features

✅ Rewritten Question Viewer

✅ Retrieved Chunk Viewer

✅ Page Metadata Inspection

✅ Retrieval Debugging Sidebar

These features help analyze exactly what the retriever is sending to the LLM.

---

# 🏗️ Architecture

```text
User Uploads PDF
        │
        ▼
PDF Loader
        │
        ▼
Text Chunking
        │
        ▼
Embeddings
(all-MiniLM-L6-v2)
        │
        ▼
FAISS Vector Store
        │
        ▼
History-Aware Retrieval
        │
        ▼
Gemini 2.5 Flash
        │
        ▼
Answer Generation
        │
        ▼
Streamlit Chat UI
```

---

# 📂 Project Structure

```text
RAG_Pdf_reader/
│
├── streamlit_app.py
│
├── src/
│   ├── loader.py
│   ├── model.py
│   ├── vectorstore.py
│   ├── rag_chain.py
│   ├── history_aware_retriever.py
│   ├── debug_retriever.py
│   └── chat_memory.py
│
├── data/
│
├── faiss_index/
│
├── .env
├── requirements.txt
└── README.md
```

---

# ⚙️ Technologies Used

## Frontend

* Streamlit

## LLM

* Gemini 2.5 Flash

## Embeddings

* sentence-transformers/all-MiniLM-L6-v2

## Vector Database

* FAISS

## Framework

* LangChain

## PDF Processing

* PyPDFLoader

---

# 🛠️ Installation

## Clone Repository

```bash
git clone <repository-url>
cd RAG_Pdf_reader
```

---

## Create Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Gemini API Key

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
```

Get your API key from:

https://aistudio.google.com/app/apikey

---

# 🚀 Run Application

```bash
python -m streamlit run streamlit_app.py
```

Open your browser:

```text
http://localhost:8501
```

---

# 💬 How to Use

## Step 1

Upload a PDF using the sidebar.

---

## Step 2

Wait for the application to:

* Load PDF
* Create Chunks
* Generate Embeddings
* Build FAISS Index

---

## Step 3

Ask questions

Example:

```text
What is Linear Regression?
```

```text
Explain Logistic Regression.
```

```text
What are the assumptions of Linear Regression?
```

---

## Step 4

Ask follow-up questions

```text
What are its examples?
```

```text
Explain it in simple words.
```

The system automatically rewrites the question before retrieval.

---

# 📊 Current Configuration

## Chunking

```python
chunk_size = 500
chunk_overlap = 50
```

### Why?

* Better retrieval quality
* Lower embedding cost
* More precise chunk matching

---

## Retrieval

```python
k = 4
```

Top 4 most relevant chunks are retrieved.

---

# 🔍 Debug Panel

The sidebar displays:

## Rewritten Question

Example:

```text
What are examples of Linear Regression?
```

---

## Retrieved Chunks

Displays:

* File Name
* Page Number
* Chunk Content

Useful for:

* Retrieval debugging
* RAG evaluation
* Hallucination analysis

---

# 🐛 Common Issues

## pypdf Not Found

```bash
pip install pypdf
```

---

## Streamlit Using Wrong Environment

Always run:

```bash
python -m streamlit run streamlit_app.py
```

instead of:

```bash
streamlit run streamlit_app.py
```

---

## Missing Torch Dependencies

```bash
pip install torch torchvision sentence-transformers
```

---

## API Key Error

Verify your `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
```

---

# 📈 Future Enhancements

## Planned Features

* Multi PDF Upload
* Streaming Responses
* Hybrid Search (BM25 + Vector Search)
* Reranking
* Programmatic Citations
* LangGraph Integration
* Agentic RAG
* Web Search Integration
* Persistent Chat History
* User Authentication

---

# 🎯 Learning Outcomes

This project demonstrates:

* Retrieval-Augmented Generation (RAG)
* Vector Databases (FAISS)
* Embeddings
* Semantic Search
* LangChain Pipelines
* Conversational AI
* History-Aware Retrieval
* Streamlit Development
* Gemini API Integration

---

# 📄 License

This project is intended for learning, experimentation, and portfolio purposes.

---

# 👨‍💻 Author

Built as part of a hands-on Generative AI and RAG learning journey using LangChain, Gemini, FAISS, and Streamlit.
