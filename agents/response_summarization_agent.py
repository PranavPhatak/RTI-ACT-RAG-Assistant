import os

from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate

from rag_retriever import ConversationMemory
from dotenv import load_dotenv
load_dotenv()


class ResponseSummarizationAgent:

    def __init__(self):

        self.llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0
        )

        # This agent has two jobs, so it keeps two memories:
        # earlier document summaries and earlier final answers.
        self.summary_memory = ConversationMemory(
            max_turns=10,
            max_chars=1000
        )

        self.response_memory = ConversationMemory(
            max_turns=10,
            max_chars=1200
        )

        # ====================================================
        # DOCUMENT SUMMARIZATION PROMPT
        # ====================================================

        self.summary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a Legal RTI Document
Summarization Agent.

Summarize the uploaded RTI document
in simple language.

Identify, where available:

1. Applicant
2. Public authority
3. Date
4. Subject
5. Information requested
6. Land/property involved
7. Main issue
8. Sections explicitly mentioned
9. Important facts

IMPORTANT:

Do NOT invent information.

Do NOT assume a section was used.

If a section is not explicitly mentioned,
say:

"No specific RTI section was explicitly
mentioned in the uploaded document."

Use the conversation memory and your previous summaries
ONLY to understand what the user is referring to (for
example "the second document" or "summarize it again").
Every fact in the summary must come from the uploaded
document.

Conversation memory:

{memory}

Your previous summaries in this conversation:

{agent_history}

Query understanding:

{query_analysis}

Uploaded document:

{document}

User request:

{question}
"""
                ),
                (
                    "human",
                    "Write the summary now."
                )
            ]
        )

        # ====================================================
        # FINAL RESPONSE PROMPT
        # ====================================================

        self.final_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the final response generator
for a Legal RTI Assistant.

Prepare a clear answer to the user.

Use ONLY the information supplied
by the specialist agents.

IMPORTANT:

Do not mention specialist agents.

Do not invent facts.

If a document was uploaded, use its
analysis where relevant.

If the user requested only a summary,
do not discuss similar cases.

If the user requested sections,
clearly distinguish:

- Section explicitly mentioned
- Potentially relevant section

If a requested piece of information
is unavailable, say so.

If any input says that information came from
Wikipedia, tell the user that this part is general
background from Wikipedia and NOT an authoritative
legal source or a court case.

Use the conversation memory ONLY to understand
what the user is referring to and to stay consistent
with earlier answers. Do not take new facts from it.

Keep the answer structured and easy
to understand.

User question:

{question}

Query understanding:

{query_analysis}

Document summary:

{summary}

General RTI answer:

{general_answer}

RTI section analysis:

{eligibility}

Similar cases:

{case_analysis}

Legal reasoning:

{reasoning}

Appeal draft:

{appeal}

Conversation memory:

{memory}

Your previous final answers in this conversation:

{agent_history}

Uploaded document available:

{uploaded_document_available}
"""
                ),
                (
                    "human",
                    "Write the final answer for the user now."
                )
            ]
        )


    # ========================================================
    # DOCUMENT SUMMARIZATION
    # ========================================================

    def summarize_document(
        self,
        question,
        document_text,
        query_analysis="",
        memory="",
        session_id="default"
    ):

        if not document_text.strip():

            return (
                "No uploaded document is available "
                "to summarize."
            )


        chain = (
            self.summary_prompt
            | self.llm
        )


        response = chain.invoke(
            {
                "document":
                    document_text,

                "question":
                    question,

                "query_analysis":
                    query_analysis,

                "memory":
                    memory or "No previous interactions.",

                "agent_history":
                    self.summary_memory.as_text(
                        session_id,
                        last_n=2,
                        user_label="REQUEST",
                        assistant_label="YOUR SUMMARY",
                        empty="None yet."
                    )
            }
        )


        self.summary_memory.add(
            session_id,
            question,
            response.content
        )


        return response.content


    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    def generate_final_response(
        self,
        question,
        query_analysis="",
        summary="",
        eligibility="",
        case_analysis="",
        reasoning="",
        appeal="",
        memory="",
        uploaded_document_available=False,
        general_answer="",
        session_id="default",
        remember=True
    ):

        chain = (
            self.final_prompt
            | self.llm
        )


        response = chain.invoke(
            {
                "question":
                    question,

                "query_analysis":
                    query_analysis,

                "summary":
                    summary,

                "general_answer":
                    general_answer,

                "eligibility":
                    eligibility,

                "case_analysis":
                    case_analysis,

                "reasoning":
                    reasoning,

                "appeal":
                    appeal,

                "memory":
                    memory or "No previous interactions.",

                "agent_history":
                    self.response_memory.as_text(
                        session_id,
                        last_n=2,
                        user_label="QUESTION",
                        assistant_label="YOUR ANSWER",
                        empty="None yet."
                    ),

                "uploaded_document_available":
                    uploaded_document_available
            }
        )


        # During verification/regeneration the Orchestrator passes
        # remember=False and records only the FINAL accepted answer
        # through record().
        if remember:

            self.response_memory.add(
                session_id,
                question,
                response.content
            )


        return response.content


    # ========================================================
    # MEMORY HELPERS
    # ========================================================

    def record(self, session_id, question, answer):

        self.response_memory.add(
            session_id,
            question,
            answer
        )


    def clear_memory(self, session_id="default"):

        self.summary_memory.clear(session_id)

        self.response_memory.clear(session_id)
