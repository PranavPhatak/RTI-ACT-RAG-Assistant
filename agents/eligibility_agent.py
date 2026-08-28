from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from rag_retriever import retrieve_rti_act, format_documents


class EligibilitySectionAgent:

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
You are the Eligibility and Section Analysis Agent.

Your task is to analyze the user's RTI-related
question using ONLY the provided RTI Act context.

Determine:

1. Whether the request appears relevant
   to the RTI Act.

2. Which RTI Act section(s) are relevant.

3. Whether the cited section actually supports
   the conclusion.

4. Explain the relevant provision.

5. If the provided context does not contain
   enough information, clearly say so.

Never invent a section number.

Context:

{context}

User request:

{question}
"""
                )
            ]
        )

    def run(self, question):

        docs = retrieve_rti_act(
            question,
            k=5
        )

        context = format_documents(docs)

        chain = self.prompt | self.llm

        response = chain.invoke(
            {
                "context": context,
                "question": question
            }
        )

        return {
            "analysis": response.content,
            "documents": docs
        }