from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from agents.query_understanding_agent import QueryUnderstandingAgent
from agents.eligibility_agent import EligibilitySectionAgent
from agents.case_retrieval_agent import CaseRetrievalAgent
from agents.legal_reasoning_agent import LegalReasoningAgent
from agents.appeal_drafting_agent import AppealDraftingAgent
from agents.response_summarization_agent import ResponseSummarizationAgent
from agents.user_friendly_response_agent import UserFriendlyResponseAgent
from agents.verification_agent import VerificationAgent


class Orchestrator:

    def __init__(self):

        # ====================================================
        # INITIALIZE AGENTS
        # ====================================================

        self.query_agent = QueryUnderstandingAgent()

        self.eligibility_agent = EligibilitySectionAgent()

        self.case_agent = CaseRetrievalAgent()

        self.reasoning_agent = LegalReasoningAgent()

        self.appeal_agent = AppealDraftingAgent()

        self.response_agent = ResponseSummarizationAgent()

        self.user_friendly_agent = UserFriendlyResponseAgent()

        self.verification_agent = VerificationAgent()


        # ====================================================
        # ROUTER LLM
        # ====================================================

        self.router_llm = ChatOllama(
            model="qwen3:8b",
            temperature=0
        )


        # ====================================================
        # ROUTER PROMPT
        # ====================================================

        self.router_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are the Intelligent Router of a Legal RTI Assistant.

Your job is ONLY to decide which specialist agents
are required for the user's request.

You MUST NOT answer the user's question.

The Query Understanding Agent is ALWAYS executed
before this router.

After understanding the user's request, activate ONLY
the specialist agents that are actually required.

Multiple agents can be selected.

====================================================
AVAILABLE SPECIALIST AGENTS
====================================================

SUMMARIZATION

Use when the user asks to:

- summarize an uploaded document
- explain an uploaded RTI in simple words
- provide a short summary
- explain the contents of a document

----------------------------------------------------

CASE_SEARCH

Use ONLY when the user asks to:

- find similar cases
- find previous cases
- search land-dispute cases
- compare the current case with previous cases
- find judgments/orders/cases similar to the uploaded case

DO NOT activate CASE_SEARCH merely because
the uploaded document is about a land dispute.

----------------------------------------------------

RTI_SECTION

Use when the user asks to:

- identify RTI sections
- explain an RTI section
- determine which RTI section applies
- identify sections mentioned in an uploaded RTI
- explain why a section applies
- determine RTI eligibility/procedure

----------------------------------------------------

LEGAL_ANALYSIS

Use when the user asks to:

- analyze a legal situation
- determine legal implications
- analyze rejection of an RTI
- evaluate the legal position
- interpret the facts and applicable law
- determine possible legal action

----------------------------------------------------

APPEAL_DRAFTING

Use when the user explicitly asks to:

- draft an RTI appeal
- draft a first appeal
- draft a second appeal
- prepare an appeal letter
- write an appeal against RTI rejection

----------------------------------------------------

GENERAL_QUESTION

Use for normal RTI questions that do not require
document summarization, case search, legal analysis,
or appeal drafting.

====================================================
ROUTING EXAMPLES
====================================================

Example 1:

"Summarize my uploaded RTI."

Output:

SUMMARIZATION

----------------------------------------------------

Example 2:

"Summarize my uploaded RTI and tell me
which sections apply."

Output:

SUMMARIZATION, RTI_SECTION

----------------------------------------------------

Example 3:

"Find similar land dispute cases."

Output:

CASE_SEARCH

----------------------------------------------------

Example 4:

"Find similar cases and tell me which
RTI sections apply."

Output:

CASE_SEARCH, RTI_SECTION

----------------------------------------------------

Example 5:

"Analyze my rejected RTI."

Output:

LEGAL_ANALYSIS

----------------------------------------------------

Example 6:

"My RTI was rejected. Can I appeal?"

Output:

LEGAL_ANALYSIS, RTI_SECTION

----------------------------------------------------

Example 7:

"Draft a first appeal for me."

Output:

APPEAL_DRAFTING

----------------------------------------------------

Example 8:

"Analyze my rejection and draft an appeal."

Output:

LEGAL_ANALYSIS, RTI_SECTION, APPEAL_DRAFTING

----------------------------------------------------

Example 9:

"What is Section 6 of the RTI Act?"

Output:

GENERAL_QUESTION

----------------------------------------------------

Example 10:

"What section applies to my RTI?"

Output:

RTI_SECTION

====================================================
IMPORTANT ROUTING RULES
====================================================

1. Query Understanding is ALWAYS executed.

2. Do NOT activate SUMMARIZATION unless
   the user asks for a summary or explanation
   of a document.

3. Do NOT activate CASE_SEARCH unless the user
   asks for similar/previous cases.

4. Do NOT activate RTI_SECTION unless the user
   asks about RTI sections, applicability,
   eligibility, or procedure.

5. Do NOT activate LEGAL_ANALYSIS unless the
   user asks for legal analysis or interpretation.

6. Do NOT activate APPEAL_DRAFTING unless the
   user asks for an appeal or draft.

7. Multiple agents can be activated when the
   user asks for multiple things.

8. The presence of an uploaded document alone
   must NOT determine which agent is activated.

9. Conversation memory can be used to understand
   references such as:

   "this case", "it", "that RTI", "the above",
   or "my previous request".

10. If the user asks only for a summary,
    activate only SUMMARIZATION.

11. If the user asks only for similar cases,
    activate only CASE_SEARCH.

12. If the user asks for both similar cases
    and RTI sections, activate CASE_SEARCH
    and RTI_SECTION.

====================================================
OUTPUT FORMAT
====================================================

Return ONLY the agent names separated by commas.

Do not explain your decision.

====================================================
USER QUESTION:
{question}

====================================================
CONVERSATION MEMORY:
{memory}

====================================================
UPLOADED DOCUMENT AVAILABLE:
{document_available}
"""
                )
            ]
        )


    # ========================================================
    # ROUTING FUNCTION
    # ========================================================

    def route(
        self,
        question,
        memory="",
        document_available=False
    ):

        chain = (
            self.router_prompt
            | self.router_llm
        )

        response = chain.invoke(
            {
                "question": question,
                "memory": memory,
                "document_available": str(document_available)
            }
        )

        raw = response.content.strip().upper()


        # ====================================================
        # ALLOWED ROUTES
        # ====================================================

        allowed_routes = [
            "SUMMARIZATION",
            "CASE_SEARCH",
            "RTI_SECTION",
            "LEGAL_ANALYSIS",
            "APPEAL_DRAFTING",
            "GENERAL_QUESTION"
        ]


        # ====================================================
        # ORDER OF EXECUTION
        # ====================================================

        route_order = [
            "SUMMARIZATION",
            "CASE_SEARCH",
            "RTI_SECTION",
            "LEGAL_ANALYSIS",
            "APPEAL_DRAFTING",
            "GENERAL_QUESTION"
        ]

        routes = []

        for route_name in route_order:

            if route_name in raw:

                if route_name in allowed_routes:

                    routes.append(route_name)


        # ====================================================
        # SAFETY FALLBACK
        # ====================================================

        if not routes:

            routes = [
                "GENERAL_QUESTION"
            ]

        return routes


    # ========================================================
    # MAIN WORKFLOW
    # ========================================================

    def run(
        self,
        question,
        memory="",
        uploaded_text="",
        uploaded_documents=None
    ):

        if uploaded_documents is None:
            uploaded_documents = []


        # ====================================================
        # DOCUMENT CHECK
        # ====================================================

        document_available = bool(
            uploaded_text
            and uploaded_text.strip()
        )


        # ====================================================
        # STEP 1
        # QUERY UNDERSTANDING
        # ====================================================

        understanding_input = f"""
CURRENT USER QUESTION:

{question}

CONVERSATION MEMORY:

{memory}

UPLOADED DOCUMENT:

{
    uploaded_text
    if document_available
    else "NO DOCUMENT UPLOADED"
}
"""

        query_analysis = (
            self.query_agent.run(
                understanding_input
            )
        )


        # ====================================================
        # STEP 2
        # INTELLIGENT ROUTING
        # ====================================================

        routes = self.route(
            question=question,
            memory=memory,
            document_available=document_available
        )


        # ====================================================
        # RESULT CONTAINERS
        # ====================================================

        summary = ""

        eligibility = {
            "analysis": "",
            "documents": []
        }

        cases = {
            "analysis": "",
            "documents": []
        }

        reasoning = ""

        appeal = ""


        # ====================================================
        # STEP 3
        # SUMMARIZATION
        # ====================================================

        if "SUMMARIZATION" in routes:

            summary = (
                self.response_agent.summarize_document(
                    question=question,
                    document_text=uploaded_text,
                    query_analysis=query_analysis
                )
            )


        # ====================================================
        # STEP 4
        # RTI SECTION
        # ====================================================

        if "RTI_SECTION" in routes:

            eligibility = (
                self.eligibility_agent.run(
                    question=question,
                    document_text=uploaded_text,
                    memory=memory
                )
            )


        # ====================================================
        # STEP 5
        # CASE SEARCH
        # ====================================================

        if "CASE_SEARCH" in routes:

            cases = (
                self.case_agent.run(
                    question,
                    memory=memory
                )
            )


        # ====================================================
        # STEP 6
        # LEGAL REASONING
        # ====================================================

        if "LEGAL_ANALYSIS" in routes:

            reasoning = (
                self.reasoning_agent.run(
                    question=question,

                    rti_analysis=
                        eligibility["analysis"],

                    case_analysis=
                        cases["analysis"],

                    memory=memory,

                    document_text=uploaded_text
                )
            )


        # ====================================================
        # STEP 7
        # APPEAL DRAFTING
        # ====================================================

        if "APPEAL_DRAFTING" in routes:

            # Appeal drafting requires legal reasoning.

            if not reasoning:

                reasoning = (
                    self.reasoning_agent.run(
                        question=question,

                        rti_analysis=
                            eligibility["analysis"],

                        case_analysis=
                            cases["analysis"],

                        memory=memory,

                        document_text=uploaded_text
                    )
                )


            appeal = (
                self.appeal_agent.run(
                    question=question,
                    reasoning=reasoning,
                    document_text=uploaded_text
                )
            )


        # ====================================================
        # STEP 8
        # COLLECT SOURCES
        # ====================================================

        source_documents = []

        source_documents.extend(
            eligibility.get(
                "documents",
                []
            )
        )

        source_documents.extend(
            cases.get(
                "documents",
                []
            )
        )


        sources = []

        seen_sources = set()


        for doc in source_documents:

            metadata = getattr(
                doc,
                "metadata",
                {}
            )


            filename = metadata.get(
                "source_file",
                metadata.get(
                    "source",
                    "Unknown"
                )
            )


            # LangChain PDF pages are normally
            # zero-based.

            page = metadata.get(
                "page_number",
                metadata.get(
                    "page",
                    0
                ) + 1
            )


            key = (
                filename,
                page
            )


            if key not in seen_sources:

                sources.append(
                    {
                        "file": filename,
                        "page": page
                    }
                )

                seen_sources.add(key)


        # ====================================================
        # STEP 9
        # INITIAL RESPONSE GENERATION
        # ====================================================

        generated_answer = (
            self.response_agent.generate_final_response(
                question=question,

                query_analysis=
                    query_analysis,

                summary=
                    summary,

                eligibility=
                    eligibility["analysis"],

                case_analysis=
                    cases["analysis"],

                reasoning=
                    reasoning,

                appeal=
                    appeal,

                memory=
                    memory,

                uploaded_document_available=
                    document_available
            )
        )


        # ====================================================
        # STEP 10
        # VERIFICATION
        # ====================================================

        verification_result = (
            self.verification_agent.run(
                question=question,
                answer=generated_answer,
                context=uploaded_text,
                query_analysis=query_analysis
            )
        )


        # ====================================================
        # STEP 11
        # REGENERATION IF INCORRECT
        # ====================================================

        max_attempts = 2

        attempt = 0

        verified = False

        verification_feedback = ""


        while attempt < max_attempts:

            attempt += 1


            # ------------------------------------------------
            # Read verification result
            # ------------------------------------------------

            if isinstance(
                verification_result,
                dict
            ):

                verified = verification_result.get(
                    "verified",
                    False
                )

                verification_feedback = (
                    verification_result.get(
                        "feedback",
                        ""
                    )
                )

            else:

                verification_text = str(
                    verification_result
                ).upper()

                verified = (
                    "PASS" in verification_text
                    or
                    "VERIFIED" in verification_text
                    or
                    "CORRECT" in verification_text
                )

                verification_feedback = str(
                    verification_result
                )


            # ------------------------------------------------
            # If correct, stop regeneration
            # ------------------------------------------------

            if verified:

                break


            # ------------------------------------------------
            # If incorrect, regenerate
            # ------------------------------------------------

            regeneration_question = f"""
Original User Request:

{question}

The previous generated answer was:

{generated_answer}

A verification agent reviewed the answer and
found the following problem:

{verification_feedback}

Please regenerate the answer.

IMPORTANT:

- Correct the identified problem.
- Use ONLY the available legal/document context.
- Do not invent facts.
- Do not invent RTI sections.
- Do not invent case details.
- Keep the answer directly related to the user's request.
- If information is unavailable, clearly say so.
"""

            generated_answer = (
                self.response_agent.generate_final_response(
                    question=regeneration_question,

                    query_analysis=
                        query_analysis,

                    summary=
                        summary,

                    eligibility=
                        eligibility["analysis"],

                    case_analysis=
                        cases["analysis"],

                    reasoning=
                        reasoning,

                    appeal=
                        appeal,

                    memory=
                        memory,

                    uploaded_document_available=
                        document_available
                )
            )


            # ------------------------------------------------
            # Verify regenerated answer
            # ------------------------------------------------

            verification_result = (
                self.verification_agent.run(
                    question=question,
                    answer=generated_answer,
                    context=uploaded_text,
                    query_analysis=query_analysis
                )
            )


        # ====================================================
        # STEP 12
        # USER-FRIENDLY RESPONSE
        # ====================================================

        final_answer = (
            self.user_friendly_agent.run(
                generated_answer
            )
        )


        # ====================================================
        # STEP 13
        # RETURN RESULTS
        # ====================================================

        return {

            # ------------------------------------------------
            # Routing
            # ------------------------------------------------

            "routes":
                routes,

            "route":
                ", ".join(routes),


            # ------------------------------------------------
            # Query understanding
            # ------------------------------------------------

            "query_analysis":
                query_analysis,


            # ------------------------------------------------
            # Specialist outputs
            # ------------------------------------------------

            "summary":
                summary,

            "eligibility":
                eligibility["analysis"],

            "case_analysis":
                cases["analysis"],

            "reasoning":
                reasoning,

            "appeal_draft":
                appeal,


            # ------------------------------------------------
            # Generated answer
            # ------------------------------------------------

            "generated_answer":
                generated_answer,


            # ------------------------------------------------
            # Verification
            # ------------------------------------------------

            "verification":
                verification_result,

            "verified":
                verified,

            "verification_attempts":
                attempt,


            # ------------------------------------------------
            # Final answer
            # ------------------------------------------------

            "answer":
                final_answer,


            # ------------------------------------------------
            # Sources
            # ------------------------------------------------

            "sources":
                sources
        }