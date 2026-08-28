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
of a Legal RTI Assistant.

Understand the user's request.

Identify:

1. Main objective
2. Important entities
3. Whether an uploaded document is relevant
4. References such as "this", "it", "my case"
5. What the user expects as the answer

Do NOT answer the legal question.

Provide a concise analysis.

Current request:

{question}
"""
                )
            ]
        )


    def run(self, question):

        chain = (
            self.prompt
            | self.llm
        )

        response = chain.invoke(
            {
                "question": question
            }
        )

        return response.content