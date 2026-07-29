import streamlit as st
from rag import rag_chain

# ---------- Page Config ----------
st.set_page_config(
    page_title="RTI Act Assistant",
    page_icon="📜",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS ----------
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
        color: #1f2937 !important;
    }
    .answer-box, .answer-box * {
        color: #1f2937 !important;
    }
    .stButton>button {
        background-color: #1f6feb;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1558c0;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This assistant answers questions about the "
        "**Right to Information Act, 2005** using a "
        "Retrieval-Augmented Generation (RAG) pipeline."
    )
    st.markdown("---")
    st.subheader("💡 Example Questions")
    example_questions = [
        "Who can file an RTI application?",
        "What is the time limit to respond to an RTI request?",
        "What information is exempt from disclosure?",
        "What is the fee for filing an RTI application?",
    ]
    for q in example_questions:
        if st.button(q, key=q, use_container_width=True):
            st.session_state["query_input"] = q

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()

# ---------- Session State ----------
if "history" not in st.session_state:
    st.session_state["history"] = []
if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""

# ---------- Header ----------
st.markdown('<p class="main-title">📜 RTI Act Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Ask anything about the Right to Information Act, 2005</p>',
    unsafe_allow_html=True,
)

# ---------- Input ----------
query = st.text_input(
    "Enter your question",
    value=st.session_state["query_input"],
    placeholder="e.g. How do I file an RTI application?",
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 5])
with col1:
    ask_clicked = st.button("Ask", use_container_width=True)

# ---------- Handle Query ----------
if ask_clicked:
    if query.strip():
        with st.spinner("Searching the RTI Act for your answer..."):
            try:
                answer = rag_chain(query)
            except Exception as e:
                answer = f"⚠️ Something went wrong while fetching the answer: {e}"
        st.session_state["history"].insert(0, {"question": query, "answer": answer})
        st.session_state["query_input"] = ""
    else:
        st.warning("Please enter a question before asking.")

# ---------- Display Conversation History ----------
if st.session_state["history"]:
    st.markdown("### 🗨️ Conversation")
    for i, item in enumerate(st.session_state["history"]):
        with st.expander(f"❓ {item['question']}", expanded=(i == 0)):
            st.markdown(
                f'<div class="answer-box">{item["answer"]}</div>',
                unsafe_allow_html=True,
            )
else:
    st.info("No questions asked yet. Try one of the example questions from the sidebar!")

# ---------- Footer ----------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit · Powered by RAG over the RTI Act, 2005")