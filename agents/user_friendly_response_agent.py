from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


class UserFriendlyResponseAgent:

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

The target user may have limited knowledge
of legal terminology.

Make the response understandable to a
normal person asking for help with an RTI
or land-dispute matter.

Original response:

{response}

Rewrite the response in simple,
easy-to-understand language.
"""
                )
            ]
        )


    def run(self, response):

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
                "response": response
            }
        )

        return result.content