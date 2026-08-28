import streamlit as st

from orchestrator import Orchestrator


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Legal RTI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f3a5f;
        margin-bottom: 0;
    }

    .subtitle {
        color: #5a6a7a;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }

    .answer-box {
        background-color: #f4f8fb;
        border-left: 5px solid #1f6feb;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin-top: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INITIALIZE SESSION MEMORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "case_memory" not in st.session_state:

    st.session_state.case_memory = ""


# ============================================================
# INITIALIZE ORCHESTRATOR
# ============================================================

@st.cache_resource
def get_orchestrator():

    return Orchestrator()


orchestrator = get_orchestrator()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚖️ Legal RTI Assistant")

    st.write(
        """
        This system uses a multi-agent RAG
        architecture for RTI and land-dispute
        related queries.
        """
    )

    st.markdown("---")

    st.subheader("🤖 Agents")

    st.write(
        "🔎 Query Understanding"
    )

    st.write(
        "📜 Eligibility & Section Analysis"
    )

    st.write(
        "📚 Case Retrieval"
    )

    st.write(
        "⚖️ Legal Reasoning"
    )

    st.write(
        "📝 Appeal Drafting"
    )

    st.write(
        "💬 Response Summarization"
    )

    st.markdown("---")

    st.subheader("💡 Example Questions")

    examples = [

        "What RTI section allows me to request land records?",

        "Are there previous RTI cases involving land disputes?",

        "My RTI application was rejected. What can I do?",

        "Draft a first appeal for my rejected RTI request."

    ]


    for example in examples:

        if st.button(
            example,
            use_container_width=True
        ):

            st.session_state["selected_query"] = (
                example
            )


    st.markdown("---")


    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.case_memory = ""

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<p class="main-title">'
    '⚖️ Legal RTI Assistant'
    '</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'AI-assisted RTI research for land-dispute cases'
    '</p>',
    unsafe_allow_html=True
)


# ============================================================
# DISPLAY PREVIOUS CONVERSATION
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER INPUT
# ============================================================

selected_query = st.session_state.get(
    "selected_query",
    ""
)


question = st.chat_input(
    "Ask your RTI / land-dispute question..."
)


# Use sidebar question if selected
if selected_query and not question:

    question = selected_query

    st.session_state["selected_query"] = ""


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Store user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------------
    # Build conversation memory
    # --------------------------------------------------------

    recent_messages = (
        st.session_state.messages[-10:]
    )


    memory_text = "\n".join(

        [
            f"{m['role']}: {m['content']}"
            for m in recent_messages
        ]

    )


    # --------------------------------------------------------
    # Run Multi-Agent System
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing your request..."
        ):

            try:

                result = orchestrator.run(

                    question=question,

                    memory=memory_text

                )


                answer = result["answer"]


            except Exception as e:

                answer = (
                    "⚠️ An error occurred:\n\n"
                    f"{str(e)}"
                )

                result = None


        # ----------------------------------------------------
        # Display Answer
        # ----------------------------------------------------

        st.markdown(answer)


        # ----------------------------------------------------
        # Display Sources
        # ----------------------------------------------------

        if result and result.get("sources"):

            st.markdown(
                "### 📚 Sources"
            )


            displayed = set()


            for source in result["sources"]:

                key = (
                    source["file"],
                    source["page"]
                )


                if key not in displayed:

                    st.write(
                        f"• **{source['file']}** "
                        f"— Page {source['page']}"
                    )

                    displayed.add(key)


        # ----------------------------------------------------
        # Display workflow
        # ----------------------------------------------------

        if result:

            with st.expander(
                "🔧 Agent Processing Details"
            ):

                st.write(
                    "Selected workflow:"
                )

                st.code(
                    result["route"]
                )

                st.write(
                    "Query Understanding:"
                )

                st.write(
                    result["query_analysis"]
                )


    # --------------------------------------------------------
    # Store assistant response in memory
    # --------------------------------------------------------

    st.session_state.messages.append(

        {
            "role": "assistant",
            "content": answer
        }

    )