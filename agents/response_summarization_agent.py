from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


class ResponseSummarizationAgent:

    def __init__(self):

        self.llm = ChatOllama(
            model="llama3:latest",
            temperature=0
        )


    # ========================================================
    # DOCUMENT SUMMARIZATION
    # ========================================================

    def summarize_document(
        self,
        question,
        document_text,
        query_analysis=""
    ):

        if not document_text.strip():

            return (
                "No uploaded document is available "
                "to summarize."
            )


        prompt = ChatPromptTemplate.from_messages(
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

Uploaded document:

{document}

User request:

{question}
"""
                )
            ]
        )


        chain = (
            prompt
            | self.llm
        )


        response = chain.invoke(
            {
                "document":
                    document_text,

                "question":
                    question
            }
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
        uploaded_document_available=False
    ):

        prompt = ChatPromptTemplate.from_messages(
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

Keep the answer structured and easy
to understand.

User question:

{question}

Query understanding:

{query_analysis}

Document summary:

{summary}

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

Uploaded document available:

{uploaded_document_available}
"""
                )
            ]
        )


        chain = (
            prompt
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

                "eligibility":
                    eligibility,

                "case_analysis":
                    case_analysis,

                "reasoning":
                    reasoning,

                "appeal":
                    appeal,

                "memory":
                    memory,

                "uploaded_document_available":
                    uploaded_document_available
            }
        )


        return response.content