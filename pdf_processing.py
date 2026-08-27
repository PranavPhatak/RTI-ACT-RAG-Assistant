import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


RTI_ACT_PATH = "data/RTI-Act_English.pdf"
LAND_CASES_FOLDER = "data/land_rti_cases"

VECTORSTORE_PATH = "vectorstore/land_rti_faiss"


# ============================================================
# 1. LOAD RTI ACT
# ============================================================

all_documents = []

print("\nLoading RTI Act...")

if os.path.exists(RTI_ACT_PATH):

    loader = PyPDFLoader(RTI_ACT_PATH)

    rti_documents = loader.load()

    for doc in rti_documents:
        doc.metadata["source_type"] = "RTI_ACT"
        doc.metadata["source_file"] = os.path.basename(RTI_ACT_PATH)

    all_documents.extend(rti_documents)

    print(
        f"RTI Act loaded successfully: "
        f"{len(rti_documents)} pages"
    )

else:
    print("WARNING: RTI Act PDF not found.")


# ============================================================
# 2. LOAD LAND-DISPUTE RTI CASE PDFs
# ============================================================

print("\nLoading land-dispute RTI cases...")

if not os.path.exists(LAND_CASES_FOLDER):
    raise FileNotFoundError(
        f"Folder not found: {LAND_CASES_FOLDER}"
    )


pdf_files = sorted(
    filename
    for filename in os.listdir(LAND_CASES_FOLDER)
    if filename.lower().endswith(".pdf")
)


print(f"Found {len(pdf_files)} PDF files.")


for index, filename in enumerate(pdf_files, start=1):

    pdf_path = os.path.join(
        LAND_CASES_FOLDER,
        filename
    )

    print(f"[{index}/{len(pdf_files)}] " f"Loading: {filename}")

    try:

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()

        # Add metadata
        for doc in documents:

            doc.metadata["source_type"] = "LAND_RTI_CASE"

            doc.metadata["source_file"] = filename

        all_documents.extend(documents)

        print(
            f"    Loaded {len(documents)} pages"
        )

    except Exception as e:

        print(f"ERROR loading {filename}: {e}")


# ============================================================
# 3. CHECK DOCUMENTS
# ============================================================

print("\n======================================")
print("DOCUMENT LOADING COMPLETE")
print("======================================")

print(f"Total pages/documents loaded: "f"{len(all_documents)}")


if len(all_documents) == 0:
    raise ValueError("No documents were loaded.")


# ============================================================
# 4. TEXT SPLITTING
# ============================================================

print("\nSplitting documents into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len,
)


chunks = text_splitter.split_documents(
    all_documents
)


print(
    f"Total chunks created: {len(chunks)}"
)


# ============================================================
# 5. DISPLAY SAMPLE CHUNK
# ============================================================

print("\nSample chunk:")
print("--------------------------------------")

if chunks:

    sample = chunks[0]

    print(
        sample.page_content[:500]
    )

    print("\nMetadata:")
    print(sample.metadata)


# ============================================================
# 6. CREATE EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding = OllamaEmbeddings(
    model="qwen3-embedding:0.6b"
)


# ============================================================
# 7. CREATE FAISS VECTOR DATABASE
# ============================================================

print("\nCreating FAISS vector database...")

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embedding
)


# ============================================================
# 8. CREATE VECTORSTORE DIRECTORY
# ============================================================

os.makedirs(
    VECTORSTORE_PATH,
    exist_ok=True
)


# ============================================================
# 9. SAVE FAISS DATABASE
# ============================================================

print("\nSaving FAISS database...")

vector_store.save_local(
    VECTORSTORE_PATH
)


# ============================================================
# 10. COMPLETE
# ============================================================

print("\n======================================")
print("VECTOR DATABASE CREATED SUCCESSFULLY")
print("======================================")

print(
    f"Location: {VECTORSTORE_PATH}"
)

print(
    f"Total documents/pages: {len(all_documents)}"
)

print(
    f"Total chunks: {len(chunks)}"
)

print("\nKnowledge base contains:")

print("  - RTI Act, 2005")

print(
    f"  - {len(pdf_files)} land-dispute RTI case PDFs"
)

print("\nYou can now run your Streamlit application.")