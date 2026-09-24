import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from rag_retriever import ConversationMemory
from dotenv import load_dotenv

load_dotenv()


class AppealDraftingAgent:

    def __init__(self):

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.7-flash",
            temperature=0
        )

        # This agent's own memory: previous appeal drafts.
        # Drafts are long, so a larger per-entry limit is used.
        self.memory = ConversationMemory(
            max_turns=5,
            max_chars=6000
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are an RTI Appeal Drafting Agent.

Draft an appropriate RTI appeal based only
on the provided information.

Do not invent:

- dates
- names
- application numbers
- authorities
- facts
- legal provisions

Use placeholders where information
is missing.

The draft should contain:

1. Appellate authority
2. Applicant details
3. Original RTI details
4. Grounds for appeal
5. Requested relief
6. Date/place
7. Signature placeholder

If the user asks to modify, shorten, extend or correct
the previous draft, start from PREVIOUS DRAFT and apply
ONLY the requested changes. Otherwise write a new draft.

Use the conversation memory ONLY to understand references
such as "this", "my RTI" or "the rejection". Do not take
new facts from it that the other inputs do not support.

Conversation memory:

{memory}

PREVIOUS DRAFT (your latest draft in this conversation):

{agent_history}

Uploaded document:

{document}

Legal reasoning:

{reasoning}

User request:

{question}
"""
                ),
                (
                    "human",
                    "Write the appeal draft now."
                )
            ]
        )


    def run(
        self,
        question,
        reasoning="",
        document_text="",
        memory="",
        session_id="default"
    ):

        chain = (
            self.prompt
            | self.llm
        )


        response = chain.invoke(
            {
                "question":
                    question,

                "reasoning":
                    reasoning or "Not available.",

                "document":
                    document_text
                    if document_text
                    else "No uploaded document.",

                "memory":
                    memory or "No previous interactions.",

                "agent_history":
                    self.memory.as_text(
                        session_id,
                        last_n=1,
                        user_label="REQUEST",
                        assistant_label="DRAFT",
                        empty="No previous draft."
                    )
            }
        )


        self.memory.add(
            session_id,
            question,
            response.content
        )


        return response.content


    def clear_memory(self, session_id="default"):

        self.memory.clear(session_id)
