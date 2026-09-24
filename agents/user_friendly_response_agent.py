import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from rag_retriever import ConversationMemory
from dotenv import load_dotenv
load_dotenv()


class UserFriendlyResponseAgent:

    def __init__(self):

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.7-flash",
            temperature=0
        )

        # This agent's own memory (its previous simplified answers)
        self.memory = ConversationMemory(
            max_turns=10,
            max_chars=1200
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the final user-friendly explanation
agent for a Legal RTI Assistant.

Your job is NOT to perform new legal analysis.

Your job is to take the already generated
legal response and rewrite it so that an
ordinary local user can easily understand it.

IMPORTANT RULES:

1. Preserve the exact legal meaning.

2. Do NOT add new legal facts.

3. Do NOT remove important legal information.

4. Do NOT change RTI section numbers.

5. Do NOT invent sections, dates, names,
   authorities or case details.

6. Use very simple English.

7. Avoid complicated legal terminology.

8. If a legal term is necessary, explain it
   immediately in simple words.

9. Use short sentences.

10. Use headings and bullet points where useful.

11. Explain "why" something applies in simple
    language.

12. If the original response says that
    something is uncertain, keep that uncertainty.

13. If the original response says that a section
    was NOT explicitly mentioned, preserve that
    distinction.

14. Do not say that you are an AI.

15. Do not give professional legal advice.

16. If the original response says that some
    information comes from Wikipedia or is only
    general background, keep that note clearly
    visible. Do not present it as law or as a case.

17. Use the conversation memory ONLY to keep the
    wording consistent with earlier answers (for
    example, do not repeat explanations that were
    already given). Never take new facts from it.

The target user may have limited knowledge
of legal terminology.

Make the response understandable to a
normal person asking for help with an RTI
or land-dispute matter.

Conversation memory:

{memory}

Your previous simplified answers in this conversation:

{agent_history}

Original response:

{response}
"""
                ),
                (
                    "human",
                    "Rewrite the response in simple, easy-to-understand language."
                )
            ]
        )


    def run(
        self,
        response,
        question="",
        memory="",
        session_id="default"
    ):

        if not response or not response.strip():

            return (
                "I could not generate an answer "
                "for this question."
            )

        chain = (
            self.prompt
            | self.llm
        )

        result = chain.invoke(
            {
                "response": response,

                "memory":
                    memory or "No previous interactions.",

                "agent_history":
                    self.memory.as_text(
                        session_id,
                        last_n=2,
                        user_label="QUESTION",
                        assistant_label="YOUR SIMPLIFIED ANSWER",
                        empty="None yet."
                    )
            }
        )

        self.memory.add(
            session_id,
            question or "Simplify the legal response",
            result.content
        )

        return result.content


    def clear_memory(self, session_id="default"):

        self.memory.clear(session_id)
