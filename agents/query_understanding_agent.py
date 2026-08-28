from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


class QueryUnderstandingAgent:

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
You are the Query Understanding Agent
for a Legal RTI Assistant specializing
in land-dispute-related RTI matters.

Analyze the user's request.

Extract:

1. Main intent
2. Type of RTI issue
3. Land-dispute subject
4. Important facts
5. Important dates
6. Authorities mentioned
7. Keywords
8. Whether the user appears to be asking for:
   - RTI information
   - RTI eligibility
   - RTI section
   - case comparison
   - appeal
   - complaint
   - appeal drafting
   - general explanation

Do NOT provide legal advice.

Return a structured analysis.

User request:

{question}
"""
                )
            ]
        )

    def run(self, question):

        chain = self.prompt | self.llm

        response = chain.invoke(
            {
                "question": question
            }
        )

        return response.content