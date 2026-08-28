import os
import tempfile

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader

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
# CSS
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

    .upload-box {
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #d0d7de;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []


if "uploaded_text" not in st.session_state:
    st.session_state.uploaded_text = ""


if "selected_query" not in st.session_state:
    st.session_state.selected_query = ""


# ============================================================
# ORCHESTRATOR
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
        AI-assisted legal research for
        RTI and land-dispute cases.
        """
    )

    # ========================================================
    # FILE UPLOAD SECTION
    # ========================================================

    st.markdown("---")

    st.subheader("📄 Upload RTI Document")

    st.caption(
        "Optional: Upload an RTI, rejection letter, "
        "appeal, or other case document."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        key="rti_uploader"
    )

    if uploaded_file is not None:

        existing_files = {
            item["name"]
            for item in st.session_state.uploaded_documents
        }

        if uploaded_file.name not in existing_files:

            with st.spinner(
                "Reading uploaded document..."
            ):

                temp_path = None

                try:

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(
                            uploaded_file.getbuffer()
                        )

                        temp_path = temp_file.name


                    loader = PyPDFLoader(
                        temp_path
                    )

                    documents = loader.load()


                    # ----------------------------------------
                    # Preserve page information
                    # ----------------------------------------

                    document_text_parts = []

                    for doc in documents:

                        page_number = (
                            doc.metadata.get(
                                "page",
                                0
                            ) + 1
                        )

                        document_text_parts.append(
                            f"""
[FILE: {uploaded_file.name}
[PAGE: {page_number}]

{doc.page_content}
"""
                        )


                    document_text = "\n".join(
                        document_text_parts
                    )


                    # ----------------------------------------
                    # Store document
                    # ----------------------------------------

                    st.session_state.uploaded_documents.append(
                        {
                            "name": uploaded_file.name,
                            "pages": len(documents)
                        }
                    )


                    st.session_state.uploaded_text += (
                        "\n\n"
                        + document_text
                    )


                    # ----------------------------------------
                    # Display upload message
                    # ----------------------------------------

                    st.session_state.messages.append(
                        {
                            "role": "system",
                            "content":
                                f"📄 Uploaded: "
                                f"**{uploaded_file.name}** "
                                f"({len(documents)} pages)"
                        }
                    )


                    st.success(
                        "Document uploaded successfully."
                    )

                except Exception as e:

                    st.error(
                        f"Could not read PDF: {e}"
                    )

                finally:

                    if (
                        temp_path
                        and os.path.exists(temp_path)
                    ):

                        os.remove(temp_path)


    # ========================================================
    # UPLOADED DOCUMENTS
    # ========================================================

    if st.session_state.uploaded_documents:

        st.markdown("---")

        st.subheader("📚 Current Documents")

        for document in (
            st.session_state.uploaded_documents
        ):

            st.write(
                f"📄 {document['name']}"
            )

            st.caption(
                f"{document['pages']} pages"
            )


    # ========================================================
    # AGENTS
    # ========================================================

    st.markdown("---")

    st.subheader("🤖 Multi-Agent System")

    st.write("🔎 Query Understanding")
    st.write("📄 Document Summarization")
    st.write("📚 Case Retrieval")
    st.write("📜 RTI Section Analysis")
    st.write("⚖️ Legal Reasoning")
    st.write("📝 Appeal Drafting")
    st.write("💬 Response Generation")


    # ========================================================
    # EXAMPLE QUESTIONS
    # ========================================================

    st.markdown("---")

    st.subheader("💡 Examples")

    examples = [

        "What is Section 6 of the RTI Act?",

        "Summarize the uploaded RTI in simple words.",

        "Find similar land-dispute cases.",

        "Which RTI sections apply to this request?",

        "Find similar cases and tell me which RTI sections apply.",

        "My RTI was rejected. Can I file an appeal?",

        "Draft a first appeal for my rejected RTI."

    ]

    for example in examples:

        if st.button(
            example,
            use_container_width=True
        ):

            st.session_state.selected_query = example


    # ========================================================
    # CLEAR
    # ========================================================

    st.markdown("---")

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.uploaded_documents = []

        st.session_state.uploaded_text = ""

        st.session_state.selected_query = ""

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<p class="main-title">'
    '⚖️ Legal RTI Assistant'
    '</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'AI-assisted RTI and land-dispute legal research'
    '</p>',
    unsafe_allow_html=True
)


# ============================================================
# SHOW CONVERSATION
# ============================================================

for message in st.session_state.messages:

    role = message["role"]

    if role == "system":

        st.info(
            message["content"]
        )

    else:

        with st.chat_message(role):

            st.markdown(
                message["content"]
            )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask your RTI or land-dispute question..."
)


# ============================================================
# SIDEBAR EXAMPLE QUESTION
# ============================================================

if (
    st.session_state.selected_query
    and not question
):

    question = (
        st.session_state.selected_query
    )

    st.session_state.selected_query = ""


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # ========================================================
    # STORE USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    with st.chat_message("user"):

        st.markdown(question)


    # ========================================================
    # BUILD CONVERSATION MEMORY
    # ========================================================

    conversation_messages = (
        st.session_state.messages[-12:]
    )


    conversation_memory = "\n".join(

        f"{message['role'].upper()}: "
        f"{message['content']}"

        for message in conversation_messages
    )


    # ========================================================
    # RUN ORCHESTRATOR
    # ========================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "Understanding your request..."
        ):

            try:

                result = orchestrator.run(

                    question=question,

                    memory=conversation_memory,

                    uploaded_text=(
                        st.session_state.uploaded_text
                    ),

                    uploaded_documents=(
                        st.session_state.uploaded_documents
                    )
                )


                answer = result["answer"]


            except Exception as e:

                result = None

                answer = (
                    "⚠️ Something went wrong.\n\n"
                    f"{str(e)}"
                )


        # ====================================================
        # DISPLAY ANSWER
        # ====================================================

        st.markdown(answer)


        # ====================================================
        # SOURCES
        # ====================================================

        if (
            result
            and result.get("sources")
        ):

            st.markdown(
                "### 📚 Sources"
            )

            seen = set()

            for source in result["sources"]:

                key = (
                    source.get("file"),
                    source.get("page")
                )

                if key in seen:
                    continue

                seen.add(key)

                st.write(
                    f"• **{source.get('file')}** "
                    f"— Page {source.get('page')}"
                )


        # ====================================================
        # WORKFLOW DISPLAY
        # ====================================================

        if result:

            with st.expander(
                "🔧 Agent Processing Details"
            ):

                st.write(
                    "Activated agents:"
                )

                for route in result.get(
                    "routes",
                    []
                ):

                    st.write(
                        f"✅ {route}"
                    )


    # ========================================================
    # STORE ASSISTANT RESPONSE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )