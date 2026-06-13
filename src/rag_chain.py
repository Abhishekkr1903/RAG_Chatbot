from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser


# =====================================================
# Convert retrieved documents into a single context string
# =====================================================
def format_docs(docs):

    context = []

    for doc in docs:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        context.append(
            f"""
Source: {source}
Page: {page}

Content:
{doc.page_content}
"""
        )

    return "\n\n".join(context)


# =====================================================
# Build RAG Chain
# =====================================================
def build_chain(retriever, llm):

    # ---------------------------------
    # Prompt
    # ---------------------------------
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful assistant.

If the user greets you (Hi, Hello, Hey),
respond normally.

For document questions,
answer only from the retrieved context.
Answer ONLY using the provided context.

If the answer is not found in the context,
reply with:

I don't know.

At the end of the answer,
mention the source pages used.
"""
            ),
            (
                "human",
                """
Chat History:
{history}

Question:
{question}

Context:
{context}
"""
            )
        ]
    )

    # ---------------------------------
    # Retrieve context from FAISS
    # ---------------------------------
    parallel = RunnableParallel(
        {
            "context":
                itemgetter("question")
                | retriever
                | RunnableLambda(format_docs),

            "question":
                itemgetter("question"),

            "history":
                itemgetter("history")
        }
    )

    # ---------------------------------
    # Full Chain
    # ---------------------------------
    chain = (
        parallel
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain