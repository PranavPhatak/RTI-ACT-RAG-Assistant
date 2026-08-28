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
You are the Legal Reasoning Agent
for a Legal RTI Assistant.

Analyze the user's situation using
ONLY the information supplied by:

1. RTI Act analysis
2. Previous land-dispute RTI cases
3. User's original request

Your analysis should explain:

- Relevant RTI provisions
- Relevant previous cases
- Similarities between the user's situation
  and previous cases
- Differences
- Whether the previous cases provide useful
  guidance
- Possible reason for rejection, if supported
  by the context
- Whether an appeal may be relevant

IMPORTANT:

Do not invent legal provisions.

Do not claim that a previous case automatically
applies to the user's situation.

Do not guarantee success.

If evidence is insufficient, explicitly say so.

RTI ACT ANALYSIS:

{rti_analysis}

PREVIOUS CASE ANALYSIS:

{case_analysis}

USER REQUEST:

{question}
"""
                )
            ]
        )

    def run(
        self,
        question,
        rti_analysis,
        case_analysis
    ):

        chain = self.prompt | self.llm

        response = chain.invoke(
            {
                "question": question,
                "rti_analysis": rti_analysis,
                "case_analysis": case_analysis
            }
        )

        return response.content