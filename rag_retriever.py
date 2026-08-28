from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# CONFIGURATION
# ============================================================

VECTORSTORE_PATH = "vectorstore/land_rti_faiss"

EMBEDDING_MODEL = "qwen3-embedding:0.6b"


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

embedding = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# ============================================================
# LOAD FAISS
# ============================================================

vector_store = FAISS.load_local(
    VECTORSTORE_PATH,
    embedding,
    allow_dangerous_deserialization=True
)


# ============================================================
# RETRIEVE FROM RTI ACT
# ============================================================

def retrieve_rti_act(query, k=5):

    docs = vector_store.similarity_search(
        query,
        k=k,
        filter={
            "source_type": "RTI_ACT"
        }
    )

    return docs


# ============================================================
# RETRIEVE LAND-DISPUTE CASES
# ============================================================

def retrieve_land_cases(query, k=5):

    docs = vector_store.similarity_search(
        query,
        k=k,
        filter={
            "source_type": "LAND_RTI_CASE"
        }
    )

    return docs


# ============================================================
# RETRIEVE FROM EVERYTHING
# ============================================================

def retrieve_all(query, k=5):

    docs = vector_store.similarity_search(
        query,
        k=k
    )

    return docs


# ============================================================
# FORMAT DOCUMENTS
# ============================================================

def format_documents(docs):

    formatted = []

    for doc in docs:

        source_file = doc.metadata.get(
            "source_file",
            "Unknown"
        )

        source_type = doc.metadata.get(
            "source_type",
            "Unknown"
        )

        page = doc.metadata.get(
            "page_number",
            doc.metadata.get("page", 0) + 1
        )

        formatted.append(
            f"""
Source Type: {source_type}
Source File: {source_file}
Page: {page}

Content:
{doc.page_content}
"""
        )

    return "\n\n".join(formatted)