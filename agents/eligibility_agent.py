import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from rag_retriever import (
    retrieve_rti_act,
    format_documents,
    ConversationMemory
)

from dotenv import load_dotenv
load_dotenv()


class EligibilitySectionAgent:

    def __init__(self):

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.7-flash",
            temperature=0
        )

        # This agent's own memory (its previous analyses)
        self.memory = ConversationMemory(
            max_turns=10,
            max_chars=1000
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
   (or, if FAISS had nothing relevant, general
   background retrieved from Wikipedia)

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

Context whose Source Type is WIKIPEDIA is general
background only. If you use it, say clearly that it
comes from Wikipedia and is NOT an authoritative
legal text. Never present Wikipedia content as the
wording of an RTI Act section. If the context says
that no relevant context was found, say so.

Use the conversation memory ONLY to understand
references such as "this", "it" or "my RTI".

CONVERSATION MEMORY:

{memory}

YOUR PREVIOUS ANALYSES IN THIS CONVERSATION
(for continuity only):

{agent_history}

USER QUESTION:

{question}

UPLOADED DOCUMENT:

{document}

RTI ACT CONTEXT:

{rti_context}
"""
                ),
                (
                    "human",
                    "Provide your RTI section analysis now."
                )
            ]
        )


    def run(
        self,
        question,
        document_text="",
        memory="",
        session_id="default",
        search_query=None
    ):

        # FAISS first; Wikipedia only if FAISS has nothing relevant
        docs = retrieve_rti_act(
            search_query or question,
            k=5
        )


        rti_context = (
            format_documents(docs)
            if docs
            else "NO RELEVANT RTI ACT CONTEXT WAS FOUND "
                 "IN THE KNOWLEDGE BASE OR WIKIPEDIA."
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
                "rti_context": rti_context,
                "memory": memory or "No previous interactions.",
                "agent_history": self.memory.as_text(
                    session_id,
                    last_n=2,
                    user_label="QUESTION",
                    assistant_label="YOUR ANALYSIS",
                    empty="None yet."
                )
            }
        )


        self.memory.add(
            session_id,
            question,
            response.content
        )


        return {

            "analysis":
                response.content,

            "documents":
                docs
        }


    def clear_memory(self, session_id="default"):

        self.memory.clear(session_id)
