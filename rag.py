from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

llm = ChatOllama(model='llama3:latest')

loader = PyPDFLoader('data/RTI-Act_English.pdf')
law_docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
chunks = text_splitter.split_documents(law_docs)

embedding = OllamaEmbeddings(model="qwen3-embedding:0.6b")
vector_store = FAISS.from_documents(documents=chunks, embedding=embedding)
retriever = vector_store.as_retriever(search_kwargs={'k':5})

prompt = ChatPromptTemplate([
    ("system", 
        """You are a Legal RTI Assistant specializing in the Right to Information Act, 2005.

        Answer the user's question using ONLY the provided legal context.

        Do not invent or assume legal provisions.

        If the answer cannot be found in the provided context, clearly state that the information could not be found in the available documents.

        Provide the relevant section or provision whenever possible.

        This system provides legal information for research purposes and does not replace professional legal advice.

        Context:
        {context}
        """
    ),
    ("user", "Question: {question}")
])


def rag_chain(question):
    docs = retriever.invoke(question)

    context = "\n".join(doc.page_content for doc in docs)

    chain = prompt | llm

    response = chain.invoke({
        'context':context,
        'question':question
    })

    return response.content

