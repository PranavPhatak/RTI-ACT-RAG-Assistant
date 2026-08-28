from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


class ResponseSummarizationAgent:

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
You are the Response Summarization Agent.

Create a clear and concise final response
for the user.

Use ONLY the information provided.

Structure the answer as:

### Answer

Direct answer to the user's question.

### Relevant RTI Provision

Mention relevant section(s), if supported
by the context.

### Similar Land-Dispute Cases

Mention relevant previous cases, if available.

### What This Means

Explain the result in simple language.

### Sources

List the source documents and page numbers
provided in the source information.

IMPORTANT:

Do not invent information.

Do not present assumptions as facts.

Do not guarantee a legal outcome.

This is legal information for research purposes
and is not a substitute for professional legal advice.

User Question:

{question}

Legal Reasoning:

{reasoning}

Case Analysis:

{case_analysis}

Source Information:

{sources}
"""
                )
            ]
        )

    def run(
        self,
        question,
        reasoning,
        case_analysis,
        sources
    ):

        chain = self.prompt | self.llm

        response = chain.invoke(
            {
                "question": question,
                "reasoning": reasoning,
                "case_analysis": case_analysis,
                "sources": sources
            }
        )

        return response.content