from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


class LegalReasoningAgent:

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
"""
                )
            ]
        )


    def run(
        self,
        question,
        rti_analysis="",
        case_analysis="",
        memory="",
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

                "rti_analysis":
                    rti_analysis,

                "case_analysis":
                    case_analysis,

                "document":
                    document_text
                    if document_text
                    else "No uploaded document.",

                "memory":
                    memory
            }
        )


        return response.content