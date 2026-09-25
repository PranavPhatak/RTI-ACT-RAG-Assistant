import os

from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate

from rag_retriever import ConversationMemory
from dotenv import load_dotenv
load_dotenv()


class LegalReasoningAgent:

    def __init__(self):

        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0
        )

        # This agent's own memory (its previous reasoning)
        self.memory = ConversationMemory(
            max_turns=10,
            max_chars=1200
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the Legal Reasoning Agent.

Analyze the user's RTI/land-dispute situation.

Use:

- RTI section analysis
- Similar case analysis
- Uploaded document
- Conversation context

Do not invent facts.

Clearly distinguish:

FACTS
LEGAL PROVISIONS
ANALYSIS
CONCLUSION

If any input says that information came from Wikipedia,
treat it only as general background and label it as such.
Never treat Wikipedia as an authoritative legal provision
or as a court case.

Use the conversation memory and your previous reasoning ONLY
for continuity (e.g. to understand "this case" or "as before").
Do not take new facts from them unless the current inputs
support those facts.

This is legal information for research
and should not be presented as professional
legal advice.

Question:

{question}

RTI analysis:

{rti_analysis}

Similar cases:

{case_analysis}

Uploaded document:

{document}

Conversation memory:

{memory}

Your previous reasoning in this conversation:

{agent_history}
"""
                ),
                (
                    "human",
                    "Provide your legal reasoning now."
                )
            ]
        )


    def run(
        self,
        question,
        rti_analysis="",
        case_analysis="",
        memory="",
        document_text="",
        session_id="default"
    ):

        chain = (
            self.prompt
            | self.llm
        )


        response = chain.invoke(
            {
                "question":
                    question,

                "rti_analysis":
                    rti_analysis or "Not requested.",

                "case_analysis":
                    case_analysis or "Not requested.",

                "document":
                    document_text
                    if document_text
                    else "No uploaded document.",

                "memory":
                    memory or "No previous interactions.",

                "agent_history":
                    self.memory.as_text(
                        session_id,
                        last_n=2,
                        user_label="QUESTION",
                        assistant_label="YOUR REASONING",
                        empty="None yet."
                    )
            }
        )


        self.memory.add(
            session_id,
            question,
            response.content
        )


        return response.content


    def clear_memory(self, session_id="default"):

        self.memory.clear(session_id)
