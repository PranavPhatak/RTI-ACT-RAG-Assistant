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
You are the Appeal Drafting Agent for a
Legal RTI Assistant.

Draft a FIRST APPEAL under the RTI framework
based ONLY on the information provided.

The draft should contain:

1. To
2. Appellant details placeholder
3. Subject
4. Reference to original RTI application
5. Facts
6. Information requested
7. Response/rejection received
8. Grounds for appeal
9. Requested relief
10. List of enclosures

Use placeholders such as:

[Applicant Name]
[Address]
[RTI Application Date]
[PIO Name]
[Application Number]

Do not invent facts.

Do not invent dates.

Do not invent case numbers.

If necessary information is missing,
use placeholders.

Clearly state that the draft should be
reviewed before submission.

USER REQUEST:

{question}

LEGAL REASONING:

{reasoning}

"""
                )
            ]
        )

    def run(
        self,
        question,
        reasoning
    ):

        chain = self.prompt | self.llm

        response = chain.invoke(
            {
                "question": question,
                "reasoning": reasoning
            }
        )

        return response.content