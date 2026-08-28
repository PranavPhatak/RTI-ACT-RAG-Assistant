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
You are the Land-Dispute Case Retrieval Agent.

Find similar land-dispute RTI cases
from the provided retrieved documents.

Compare:

- Land dispute issue
- RTI request
- Parties
- Authorities
- Important facts
- Legal provisions
- Outcome

Do not invent facts.

If no sufficiently similar case exists,
say so.

User request:

{question}

Conversation memory:

{memory}

Retrieved cases:

{cases}
"""
                )
            ]
        )


    def run(
        self,
        question,
        memory=""
    ):

        docs = retrieve_land_cases(
            question,
            k=5
        )


        context = format_documents(
            docs
        )


        chain = (
            self.prompt
            | self.llm
        )


        response = chain.invoke(
            {
                "question": question,

                "memory": memory,

                "cases": context
            }
        )


        return {

            "analysis":
                response.content,

            "documents":
                docs
        }