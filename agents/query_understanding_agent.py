import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from rag_retriever import ConversationMemory
from dotenv import load_dotenv
load_dotenv()
# The full document is not needed to understand the request.
# Sending only an excerpt saves tokens (important for Groq rate limits).
MAX_DOCUMENT_CHARS = 3000


class QueryUnderstandingAgent:

    def __init__(self):

        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0
        )

        # This agent's own memory (its previous analyses)
        self.memory = ConversationMemory(
            max_turns=10,
            max_chars=800
        )

        # ====================================================
        # ANALYSIS PROMPT
        # ====================================================

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the Query Understanding Agent
of a Legal RTI Assistant.

Understand the user's request.

Identify:

1. Main objective
2. Important entities
3. Whether an uploaded document is relevant
4. References such as "this", "it", "my case"
   (resolve them using the conversation memory)
5. What the user expects as the answer

Do NOT answer the legal question.

Provide a concise analysis.

Conversation memory:

{memory}

Uploaded document (excerpt):

{document}

Your previous analyses in this conversation
(for continuity only):

{agent_history}

Current request:

{question}
"""
                ),
                (
                    "human",
                    "Provide your concise analysis now."
                )
            ]
        )

        # ====================================================
        # QUERY REWRITE PROMPT (for retrieval)
        # ====================================================

        self.rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You rewrite follow-up messages into standalone search queries.

Using the conversation memory, rewrite the user's latest message
as ONE self-contained search query. Resolve references such as
"this", "it", "that section", "my case", "the above".

Rules:

- Keep RTI section numbers, names, places and legal terms.
- Do NOT answer the question.
- Do NOT add facts that are not in the memory or the message.
- If the message is already self-contained, return it unchanged.
- Output ONLY the query on a single line (maximum 40 words).

Conversation memory:

{memory}

Latest user message:

{question}
"""
                ),
                (
                    "human",
                    "Return the standalone search query now."
                )
            ]
        )

    # ========================================================
    # ANALYZE REQUEST
    # ========================================================

    def run(
        self,
        question,
        memory="",
        document_text="",
        session_id="default"
    ):

        if document_text and document_text.strip():

            document = document_text[:MAX_DOCUMENT_CHARS]

            if len(document_text) > MAX_DOCUMENT_CHARS:

                document += "\n...[document continues]"

        else:

            document = "NO DOCUMENT UPLOADED"

        chain = (
            self.prompt
            | self.llm
        )

        response = chain.invoke(
            {
                "question": question,
                "memory": memory or "No previous interactions.",
                "document": document,
                "agent_history": self.memory.as_text(
                    session_id,
                    last_n=2,
                    user_label="REQUEST",
                    assistant_label="YOUR ANALYSIS",
                    empty="None yet."
                )
            }
        )

        analysis = response.content

        self.memory.add(
            session_id,
            question,
            analysis
        )

        return analysis

    # ========================================================
    # REWRITE FOLLOW-UP INTO STANDALONE SEARCH QUERY
    # ========================================================

    def rewrite_query(self, question, memory=""):

        try:

            chain = (
                self.rewrite_prompt
                | self.llm
            )

            response = chain.invoke(
                {
                    "question": question,
                    "memory": memory or "No previous interactions."
                }
            )

            lines = [
                line.strip().strip('"').strip("'")
                for line in str(response.content).splitlines()
                if line.strip()
            ]

            rewritten = lines[0] if lines else ""

            return rewritten[:300] if rewritten else question

        except Exception:

            return question

    # ========================================================
    # CLEAR MEMORY
    # ========================================================

    def clear_memory(self, session_id="default"):

        self.memory.clear(session_id)
