def retrieve_docs(retriever, question):

    docs = retriever.invoke(question)

    return docs