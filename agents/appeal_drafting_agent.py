from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


class AppealDraftingAgent:

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

Uploaded document:

{document}

Legal reasoning:

{reasoning}

User request:

{question}
"""
                )
            ]
        )


    def run(
        self,
        question,
        reasoning="",
        document_text=""
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
                    reasoning,

                "document":
                    document_text
                    if document_text
                    else "No uploaded document."
            }
        )


        return response.content