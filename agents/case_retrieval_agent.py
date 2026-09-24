import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

from rag_retriever import (
    retrieve_land_cases,
    format_documents,
    ConversationMemory
)

class CaseRetrievalAgent:

    def __init__(self):

        # Groq LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.7-flash",
            temperature=0
        )

        # This agent's own memory (its previous case analyses)
        self.memory = ConversationMemory(
            max_turns=10,
            max_chars=1000
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
9. Documents whose Source Type is WIKIPEDIA are general background
   articles retrieved because the case database had nothing relevant.
   They are NOT cases. Never present a Wikipedia article as a case,
   judgment or order. If ONLY Wikipedia documents were retrieved,
   say: "No sufficiently similar case was found in the retrieved
   documents." and then, if useful, add a short section called
   "GENERAL BACKGROUND (Wikipedia, not a legal authority)".
10. If the retrieved documents section says that nothing relevant was
    found, use the sentence from rule 6.
11. Use the conversation memory ONLY to understand references such as
    "this case" or "it". Do not take case facts from the memory.

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

YOUR PREVIOUS CASE ANALYSES IN THIS CONVERSATION (for continuity only):
{agent_history}

USER QUERY:
{question}

CONVERSATION MEMORY:
{memory}

RETRIEVED CASES:
{cases}
"""
                ),
                (
                    "human",
                    "Provide your case analysis now."
                )
            ]
        )

    def run(
        self,
        question,
        memory="",
        session_id="default",
        search_query=None
    ):

        # Retrieve similar land-dispute cases from FAISS.
        # Wikipedia is used only if FAISS has nothing relevant.
        docs = retrieve_land_cases(
            search_query or question,
            k=5
        )

        # Format retrieved documents for the LLM
        context = (
            format_documents(docs)
            if docs
            else "NO RELEVANT CASES OR BACKGROUND DOCUMENTS "
                 "WERE FOUND IN THE KNOWLEDGE BASE OR WIKIPEDIA."
        )

        # Create LCEL chain
        chain = (
            self.prompt
            | self.llm
        )

        # Invoke LLM
        response = chain.invoke(
            {
                "question": question,
                "memory": memory or "No previous interactions.",
                "cases": context,
                "agent_history": self.memory.as_text(
                    session_id,
                    last_n=2,
                    user_label="QUERY",
                    assistant_label="YOUR ANALYSIS",
                    empty="None yet."
                )
            }
        )

        self.memory.add(
            session_id,
            question,
            response.content
        )

        return {
            "analysis": response.content,
            "documents": docs
        }

    def clear_memory(self, session_id="default"):

        self.memory.clear(session_id)
