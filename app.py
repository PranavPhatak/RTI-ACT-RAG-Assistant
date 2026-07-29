import streamlit as st
from rag import rag_chain

st.title("RTI-ACT-RAG-Assistant")

st.write("Ask questions about the Right to Information Act, 2005.")

query = st.text_input("Enter your question")

if st.button("Tell Me Answer"):
    if query:
        with st.spinner("Finding Answer..."):
            answer = rag_chain(query)
        st.subheader("Answer")
        st.write(answer)
    else:
        st.write("Please enter a question")