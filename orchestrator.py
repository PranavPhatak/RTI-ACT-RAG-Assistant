from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from agents.query_understanding_agent import (
    QueryUnderstandingAgent
)

from agents.eligibility_agent import (
    EligibilitySectionAgent
)

from agents.case_retrieval_agent import (
    CaseRetrievalAgent
)

from agents.legal_reasoning_agent import (
    LegalReasoningAgent
)

from agents.appeal_drafting_agent import (
    AppealDraftingAgent
)

from agents.response_summarization_agent import (
    ResponseSummarizationAgent
)


class Orchestrator:

    def __init__(self):

        # ----------------------------------------------------
        # Initialize agents
        # ----------------------------------------------------

        self.query_agent = (
            QueryUnderstandingAgent()
        )

        self.eligibility_agent = (
            EligibilitySectionAgent()
        )

        self.case_agent = (
            CaseRetrievalAgent()
        )

        self.reasoning_agent = (
            LegalReasoningAgent()
        )

        self.appeal_agent = (
            AppealDraftingAgent()
        )

        self.summary_agent = (
            ResponseSummarizationAgent()
        )


        # ----------------------------------------------------
        # Routing LLM
        # ----------------------------------------------------

        self.router_llm = ChatOllama(
            model="llama3:latest",
            temperature=0
        )


        self.router_prompt = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
You are the Orchestrator Agent.

Determine which workflow is appropriate.

Possible workflows:

1. QUESTION
   General question about RTI or land-dispute RTI.

2. APPEAL
   User wants help preparing or understanding
   an RTI appeal.

3. CASE_SEARCH
   User specifically wants similar previous
   RTI land-dispute cases.

4. DOCUMENT_ANALYSIS
   User wants analysis of an RTI/rejection
   document.

Return ONLY one of:

QUESTION
APPEAL
CASE_SEARCH
DOCUMENT_ANALYSIS

User request:

{question}
"""
                    )
                ]
            )
        )


    # ========================================================
    # ROUTER
    # ========================================================

    def route(self, question):

        chain = (
            self.router_prompt
            | self.router_llm
        )

        response = chain.invoke(
            {
                "question": question
            }
        )

        route = response.content.strip()

        allowed_routes = [
            "QUESTION",
            "APPEAL",
            "CASE_SEARCH",
            "DOCUMENT_ANALYSIS"
        ]

        for r in allowed_routes:

            if r in route:

                return r

        return "QUESTION"


    # ========================================================
    # MAIN WORKFLOW
    # ========================================================

    def run(
        self,
        question,
        memory=""
    ):

        # ----------------------------------------------------
        # STEP 1
        # Query Understanding
        # ----------------------------------------------------

        query_analysis = (
            self.query_agent.run(question)
        )


        # ----------------------------------------------------
        # STEP 2
        # Determine workflow
        # ----------------------------------------------------

        route = self.route(question)


        # ----------------------------------------------------
        # STEP 3
        # Eligibility / Section Analysis
        # ----------------------------------------------------

        eligibility = (
            self.eligibility_agent.run(question)
        )


        # ----------------------------------------------------
        # STEP 4
        # Case Retrieval
        # ----------------------------------------------------

        cases = (
            self.case_agent.run(question)
        )


        # ----------------------------------------------------
        # STEP 5
        # Legal Reasoning
        # ----------------------------------------------------

        reasoning = (
            self.reasoning_agent.run(

                question=question,

                rti_analysis=eligibility[
                    "analysis"
                ],

                case_analysis=cases[
                    "analysis"
                ]
            )
        )


        # ----------------------------------------------------
        # STEP 6
        # Appeal Drafting
        # ----------------------------------------------------

        appeal_draft = None


        if route == "APPEAL":

            appeal_draft = (
                self.appeal_agent.run(

                    question=question,

                    reasoning=reasoning
                )
            )


        # ----------------------------------------------------
        # STEP 7
        # Collect sources
        # ----------------------------------------------------

        source_documents = []

        source_documents.extend(
            eligibility["documents"]
        )

        source_documents.extend(
            cases["documents"]
        )


        sources = []


        seen = set()


        for doc in source_documents:

            filename = doc.metadata.get(
                "source_file",
                "Unknown"
            )

            page = doc.metadata.get(
                "page_number",
                doc.metadata.get(
                    "page",
                    0
                ) + 1
            )

            key = (
                filename,
                page
            )


            if key not in seen:

                sources.append(
                    {
                        "file": filename,
                        "page": page
                    }
                )

                seen.add(key)


        # ----------------------------------------------------
        # STEP 8
        # Final Response
        # ----------------------------------------------------

        final_answer = (
            self.summary_agent.run(

                question=question,

                reasoning=reasoning,

                case_analysis=cases[
                    "analysis"
                ],

                sources=sources
            )
        )


        # ----------------------------------------------------
        # Add appeal draft if requested
        # ----------------------------------------------------

        if appeal_draft:

            final_answer += (

                "\n\n"
                "## 📝 Draft First Appeal\n\n"
                + appeal_draft
            )


        # ----------------------------------------------------
        # Return complete result
        # ----------------------------------------------------

        return {

            "route": route,

            "query_analysis":
                query_analysis,

            "eligibility":
                eligibility["analysis"],

            "case_analysis":
                cases["analysis"],

            "reasoning":
                reasoning,

            "appeal_draft":
                appeal_draft,

            "answer":
                final_answer,

            "sources":
                sources
        }