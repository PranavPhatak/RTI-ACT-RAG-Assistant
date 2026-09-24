import os
import json
import re

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from rag_retriever import ConversationMemory
from dotenv import load_dotenv
load_dotenv()

class VerificationAgent:

    def __init__(self):

        # ====================================================
        # LLM
        # ====================================================

        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0
        )

        # ====================================================
        # MEMORY (previous verdicts, for consistency only)
        # ====================================================

        self.memory = ConversationMemory(
            max_turns=10,
            max_chars=500
        )

        # ====================================================
        # VERIFICATION PROMPT
        # ====================================================

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the Verification Agent of a Legal RTI Assistant.

Your job is to CHECK whether the generated answer is
supported by the available context.

You MUST NOT generate a new answer.

Check the following:

1. Is the answer relevant to the user's question?

2. Are the facts supported by the provided context?

3. Are RTI sections or provisions supported by the
   provided context?

4. Has the answer invented facts?

5. Has the answer invented case details?

6. Does the answer make unsupported legal claims?

7. If an uploaded document is available, does the
   answer correctly represent that document?

8. If the context contains WIKIPEDIA sources, does the
   answer present that information only as general
   background, and NOT as the wording of an RTI Act
   section, a binding legal rule or a court case?

IMPORTANT:

For legal information, do not assume that something
is correct merely because it sounds legally reasonable.

If the available context does not support a claim,
consider that claim unsupported.

If the answer contains information that cannot be
verified from the provided context, mark it as false.

The previous verification results are shown ONLY so that
your judgments stay consistent. Judge the CURRENT answer
ONLY against the CURRENT context.

Return ONLY valid JSON in this exact format:

{{
    "verified": true,
    "feedback": "The answer is supported by the available context."
}}

OR:

{{
    "verified": false,
    "feedback": "Explain exactly what is incorrect or unsupported."
}}

Do not include markdown.

Do not include ```json.

Do not provide a new answer.

====================================================
PREVIOUS VERIFICATION RESULTS (consistency only)
====================================================

{agent_history}

====================================================
USER QUESTION
====================================================

{question}

====================================================
QUERY UNDERSTANDING
====================================================

{query_analysis}

====================================================
AVAILABLE CONTEXT
====================================================

{context}

====================================================
GENERATED ANSWER
====================================================

{answer}

"""
                ),
                (
                    "human",
                    "Return the JSON verification result now."
                )
            ]
        )


    # ========================================================
    # VERIFY RESPONSE
    # ========================================================

    def run(
        self,
        question,
        answer,
        context="",
        query_analysis="",
        session_id="default"
    ):

        # ====================================================
        # CREATE CHAIN
        # ====================================================

        chain = self.prompt | self.llm


        # ====================================================
        # CALL LLM
        # ====================================================

        response = chain.invoke(
            {
                "question": question,

                "answer": answer,

                "context":
                    context
                    if context
                    else "NO ADDITIONAL CONTEXT AVAILABLE",

                "query_analysis":
                    query_analysis,

                "agent_history":
                    self.memory.as_text(
                        session_id,
                        last_n=3,
                        user_label="QUESTION",
                        assistant_label="RESULT",
                        empty="None yet."
                    )
            }
        )


        # ====================================================
        # GET RAW RESPONSE
        # ====================================================

        raw = str(response.content).strip()


        # ====================================================
        # REMOVE QWEN3 THINKING BLOCK
        # ====================================================

        # Qwen3 on Groq may return its reasoning inside
        # <think> ... </think> before the JSON.

        raw = re.sub(
            r"<think>.*?</think>",
            "",
            raw,
            flags=re.DOTALL | re.IGNORECASE
        ).strip()


        # ====================================================
        # REMOVE MARKDOWN CODE BLOCKS
        # ====================================================

        # Handles:
        #
        # ```json
        # {...}
        # ```
        #
        # or:
        #
        # ```
        # {...}
        # ```

        raw = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw,
            flags=re.IGNORECASE
        )

        raw = re.sub(
            r"\s*```$",
            "",
            raw
        )

        raw = raw.strip()


        # ====================================================
        # HANDLE EXTRA TEXT AROUND JSON
        # ====================================================

        # Sometimes the LLM may return something like:
        #
        # Here is the result:
        # {"verified": true, ...}

        # Try to extract the JSON object.

        json_match = re.search(
            r"\{.*\}",
            raw,
            flags=re.DOTALL
        )

        if json_match:

            raw_json = json_match.group(0)

        else:

            raw_json = raw


        # ====================================================
        # PARSE JSON
        # ====================================================

        try:

            result = json.loads(raw_json)


            verified = result.get(
                "verified",
                False
            )

            feedback = result.get(
                "feedback",
                ""
            )


            outcome = {
                "verified": bool(verified),

                "feedback": str(feedback)
            }


        except json.JSONDecodeError:

            # =================================================
            # FALLBACK
            # =================================================

            upper_raw = raw.upper()


            # Look for explicit negative verification first.

            if (
                '"VERIFIED": FALSE' in upper_raw
                or
                "VERIFIED: FALSE" in upper_raw
                or
                "INCORRECT" in upper_raw
                or
                "UNSUPPORTED" in upper_raw
                or
                "HALLUCINATION" in upper_raw
            ):

                outcome = {
                    "verified": False,

                    "feedback":
                        "The verification agent found that the generated response may contain unsupported or incorrect information."
                }


            # Look for explicit positive verification.

            elif (
                '"VERIFIED": TRUE' in upper_raw
                or
                "VERIFIED: TRUE" in upper_raw
                or
                "CORRECT" in upper_raw
                or
                "SUPPORTED" in upper_raw
            ):

                outcome = {
                    "verified": True,

                    "feedback":
                        "The verification agent marked the response as supported."
                }


            # =================================================
            # SAFEST DEFAULT
            # =================================================

            else:

                outcome = {
                    "verified": False,

                    "feedback":
                        "The verification agent did not return a valid verification result."
                }


        # ====================================================
        # REMEMBER THIS VERDICT
        # ====================================================

        self.memory.add(
            session_id,
            question,
            f"verified={outcome['verified']}; "
            f"feedback={outcome['feedback']}"
        )


        return outcome


    def clear_memory(self, session_id="default"):

        self.memory.clear(session_id)
