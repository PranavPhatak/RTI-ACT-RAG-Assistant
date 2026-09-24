import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from rag_retriever import (
    retrieve_rti_act,
    format_documents,
    ConversationMemory
)

load_dotenv()

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)


# ============================================================
# GENERAL RTI QUESTION AGENT
# ============================================================

class GeneralRAGAgent:
    """
    Answers normal RTI questions (e.g. "What is Section 6 of the
    RTI Act?") using the RTI Act chunks stored in the shared FAISS
    database. If FAISS has nothing relevant, Wikipedia is searched.
    """

    def __init__(self):

        self.llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0
        )

        # This agent's own memory (its previous answers)
        self.memory = ConversationMemory(
            max_turns=10,
            max_chars=1200
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a Legal RTI Assistant specializing in the Right to Information Act, 2005.

Answer the user's question using ONLY the provided context.

Do not invent or assume legal provisions.

If the answer cannot be found in the provided context, clearly state that the information could not be found in the available documents.

Provide the relevant section or provision whenever possible.

Context whose Source Type is WIKIPEDIA is general background
information only. If you use it, say clearly that it comes from
Wikipedia and is not an authoritative legal text. Never present
Wikipedia content as a section of the RTI Act or as a court case.

Use the conversation memory ONLY to understand references such as
"this", "it", "that section" or "the above". Never take legal facts
from the memory unless they are also supported by the context.

This system provides legal information for research purposes and does not replace professional legal advice.

Conversation memory:
{memory}

Your previous answers in this conversation:
{agent_history}

Context:
{context}
"""
                ),
                (
                    "human",
                    "Question: {question}"
                )
            ]
        )

    # --------------------------------------------------------

    def run(
        self,
        question,
        memory="",
        session_id="default",
        search_query=None
    ):

        docs = retrieve_rti_act(
            search_query or question,
            k=5
        )

        context = (
            format_documents(docs)
            if docs
            else "NO RELEVANT CONTEXT WAS FOUND IN THE "
                 "KNOWLEDGE BASE OR WIKIPEDIA."
        )

        chain = self.prompt | self.llm

        response = chain.invoke(
            {
                "question": question,
                "context": context,
                "memory": memory or "No previous interactions.",
                "agent_history": self.memory.as_text(
                    session_id,
                    last_n=2,
                    user_label="QUESTION",
                    assistant_label="YOUR ANSWER",
                    empty="None yet."
                )
            }
        )

        answer = response.content

        self.memory.add(
            session_id,
            question,
            answer
        )

        return {
            "analysis": answer,
            "documents": docs
        }

    # --------------------------------------------------------

    def clear_memory(self, session_id="default"):

        self.memory.clear(session_id)


# ============================================================
# BACKWARD-COMPATIBLE HELPER
# ============================================================

_default_agent = None


def rag_chain(question, memory="", session_id="default"):
    """
    Same entry point as the old rag.py: returns the answer text.
    """

    global _default_agent

    if _default_agent is None:

        _default_agent = GeneralRAGAgent()

    return _default_agent.run(
        question,
        memory=memory,
        session_id=session_id
    )["analysis"]
