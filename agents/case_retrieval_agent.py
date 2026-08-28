from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from rag_retriever import (
    retrieve_land_cases,
    format_documents
)


class CaseRetrievalAgent:

    def __init__(self):

        self.llm = ChatOllama(
            model="llama3:latest",
            temperature=0
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the Case Retrieval Agent for a
Legal RTI Assistant.

You are given retrieved previous
land-dispute-related RTI cases.

Identify cases that are relevant
to the user's situation.

For each relevant case, identify:

- Case/document
- Applicant/Appellant/Complainant
- Respondent/public authority
- Type of proceeding
- Main land-dispute issue
- RTI issue
- Important reasoning
- Outcome

ONLY use information contained in the
provided context.

Do not invent missing case details.

Context:

{context}

User request:

{question}
"""
                )
            ]
        )

    def run(self, question):

        docs = retrieve_land_cases(
            question,
            k=5
        )

        context = format_documents(docs)

        chain = self.prompt | self.llm

        response = chain.invoke(
            {
                "context": context,
                "question": question
            }
        )

        return {
            "analysis": response.content,
            "documents": docs
        }