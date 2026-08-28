from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from rag_retriever import (
    retrieve_rti_act,
    format_documents
)


class EligibilitySectionAgent:

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
You are the RTI Section Analysis Agent.

Analyze the user's RTI question/document.

You have two sources:

1. Uploaded document
2. RTI Act knowledge retrieved from FAISS

IMPORTANT:

Clearly distinguish:

A. Sections explicitly mentioned
   in the uploaded document.

B. Potentially relevant RTI Act sections.

Never claim that a section was used in
the uploaded document unless the document
actually mentions it.

If no section is explicitly mentioned,
say so.

Explain why a potentially relevant
section may apply.

Use only the provided context.

USER QUESTION:

{question}

UPLOADED DOCUMENT:

{document}

RTI ACT CONTEXT:

{rti_context}
"""
                )
            ]
        )


    def run(
        self,
        question,
        document_text="",
        memory=""
    ):

        docs = retrieve_rti_act(
            question,
            k=5
        )


        rti_context = format_documents(
            docs
        )


        document = (
            document_text
            if document_text.strip()
            else "No uploaded document."
        )


        chain = (
            self.prompt
            | self.llm
        )


        response = chain.invoke(
            {
                "question": question,
                "document": document,
                "rti_context": rti_context
            }
        )


        return {

            "analysis":
                response.content,

            "documents":
                docs
        }