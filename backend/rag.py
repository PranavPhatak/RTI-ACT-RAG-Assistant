from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

llm = ChatOllama(model='llama3:latest')

loader = PyPDFLoader('data/RTI-ACT-English.pdf')
law_docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
chunks = text_splitter.split_documents(law_docs)

embedding = OllamaEmbeddings(model="qwen3-embedding:0.6b")
vector_store = FAISS.from_documents(documents=chunks, embedding=embedding)
retriever = vector_store.as_retriever(search_kwargs={'k':5})

prompt = ChatPromptTemplate([
    ("system", 
        """You are a helpful research paper assistant.

        Answer the user's question using ONLY the provided context.

        If the answer is not present in the context, say that you could not find the answer in the research paper.

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

