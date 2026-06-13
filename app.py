from src.loader import load_pdfs
from src.model import get_embeddings, get_llm
from src.vectorstore import create_or_load_vectorstore
from src.rag_chain import build_chain
from src.chat_memory import chat_history


PDF_FOLDER = "data"

print("Loading PDFs...")

documents = load_pdfs(PDF_FOLDER)

print(f"Loaded {len(documents)} pages")

embeddings = get_embeddings()

vectorstore = create_or_load_vectorstore(
    documents,
    embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

llm = get_llm()

chain = build_chain(
    retriever,
    llm
)

print("\nRAG Ready")
print("Type 'exit' to quit.\n")

while True:

    question = input("Q: ")

    if question.lower() == "exit":
        break

    answer = chain.invoke(
    {
        "question": question,
        "history": chat_history.messages
    }
)
    chat_history.add_user_message(question)

    chat_history.add_ai_message(answer)
    print("\nA:", answer)
    print()