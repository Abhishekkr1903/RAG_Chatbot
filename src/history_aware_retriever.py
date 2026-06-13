from langchain_core.prompts import ChatPromptTemplate


rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Given the conversation history and latest user question,
            rewrite the question so it can be understood without
            the conversation history.

            Only return the rewritten question.
            """
        ),
        (
            "human",
            """
            Chat History:
            {history}

            Question:
            {question}
            """
        )
    ]
)


def rewrite_question(llm, question, history):

    chain = rewrite_prompt | llm

    response = chain.invoke(
        {
            "history": history,
            "question": question
        }
    )

    return response.content