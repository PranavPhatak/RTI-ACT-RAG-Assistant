import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

RTI_ACT_PATH = "data/RTI-Act_English.pdf"
LAND_CASES_FOLDER = "data/land_rti_cases"
VECTORSTORE_PATH = "vectorstore/land_rti_database"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

EMBEDDING_MODEL = "qwen3-embedding:0.6b"

def load_pdf(pdf_path, source_type):
    filename = os.path.basename(pdf_path)
    print(f"Loading: {filename}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    for doc in documents:
        doc.metadata['source_type'] = source_type
        doc.metadata['source_file'] = filename
        doc.metadata['page_number'] = doc.metadata.get("page",0)+1

    return documents

print("Legal RTI Knowledge Base Creation")

if not os.path.exists(RTI_ACT_PATH):
    raise FileNotFoundError(f"\nRTI ACT pdf not found: \n{RTI_ACT_PATH}")

if not os.path.exists(LAND_CASES_FOLDER):
    raise FileNotFoundError(f"\nLand Dispute case folder not found:\n {LAND_CASES_FOLDER}")
