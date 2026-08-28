from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from rag_retriever import (
    retrieve_land_cases,
    format_documents
)


class CaseRetrievalAgent:

    def __init__(self):

        # Local LLM
        self.llm = ChatOllama(
            model="llama3:latest",
            temperature=0
        )

        # Prompt for similar case analysis
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the Land-Dispute Case Retrieval Agent in a Legal RTI Assistant.

Your task is to identify and explain the most relevant land-dispute RTI
cases from the retrieved documents.

The user has described a legal/RTI issue. You must compare the user's
issue with the retrieved cases and clearly explain WHY each selected case
is relevant.

IMPORTANT RULES:

1. Use ONLY the information available in the retrieved cases.
2. Do NOT invent facts, legal provisions, outcomes, parties, authorities,
   or case details.
3. Do NOT assume that two cases are similar just because they involve land.
4. Focus on substantive similarity between the user's query and the cases.
5. If a case is only partially similar, clearly mention the limitation.
6. If none of the retrieved cases are sufficiently similar, say:
   "No sufficiently similar case was found in the retrieved documents."
7. Distinguish between facts explicitly present in the documents and
   conclusions based on similarity.
8. Explain the relevance of every case you include.

Compare the user's query with the retrieved cases using these factors:

- Land dispute issue
- Nature of the RTI request
- Information requested
- Parties involved
- Authorities involved
- Important facts
- Legal provisions or RTI sections mentioned
- Reason for information being provided or denied
- Case outcome
- Overall similarity

For every relevant case, provide the following structure:

CASE 1

Case / Document:
[Identify the case or document using the information available.]

Similarity:
[High / Moderate / Low]

Why this case is relevant:
[Clearly explain how the facts and RTI issue in this case relate
to the user's query. This is the MOST IMPORTANT part.]

Matching points:
- [Point 1]
- [Point 2]
- [Point 3]

RTI aspect:
[Explain how the RTI request in this case is similar or different.]

Authorities:
[Authorities involved, only if available.]

Legal provisions:
[Relevant provisions mentioned in the document, only if available.]

Outcome:
[Outcome of the case, only if available.]

Differences / limitations:
[Explain important differences between the user's case and this case.]

Repeat this structure for each sufficiently relevant case.

At the end provide:

OVERALL ASSESSMENT

[Briefly explain which retrieved case is the closest match to the user's
query and why.]

USER QUERY:
{question}

CONVERSATION MEMORY:
{memory}

RETRIEVED CASES:
{cases}
"""
                )
            ]
        )

    def run(
        self,
        question,
        memory=""
    ):

        # Retrieve similar land-dispute cases
        docs = retrieve_land_cases(
            question,
            k=5
        )

        # Format retrieved documents for the LLM
        context = format_documents(docs)

        # Create LCEL chain
        chain = (
            self.prompt
            | self.llm
        )

        # Invoke LLM
        response = chain.invoke(
            {
                "question": question,
                "memory": memory,
                "cases": context
            }
        )

        return {
            "analysis": response.content,
            "documents": docs
        }