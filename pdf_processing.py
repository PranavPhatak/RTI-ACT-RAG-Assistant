import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# ------------------------------------------------------------
# RTI Act
# ------------------------------------------------------------

RTI_ACT_PATH = "data/RTI-Act_English.pdf"


# ------------------------------------------------------------
# Folder containing ALL land-dispute RTI cases
# ------------------------------------------------------------

LAND_CASES_FOLDER = "data/land_rti_cases"


# ------------------------------------------------------------
# FAISS database location
# ------------------------------------------------------------

VECTORSTORE_PATH = "vectorstore/land_rti_faiss"


# ------------------------------------------------------------
# Chunk configuration
# ------------------------------------------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# ------------------------------------------------------------
# Embedding model
# ------------------------------------------------------------

EMBEDDING_MODEL = "qwen3-embedding:0.6b"


# ============================================================
# HELPER FUNCTION
# ============================================================

def load_pdf(pdf_path, source_type):
    """
    Load a PDF and add useful metadata to every page.
    """

    filename = os.path.basename(pdf_path)

    print(f"Loading: {filename}")

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    for doc in documents:

        # Identify the type of document
        doc.metadata["source_type"] = source_type

        # Original PDF filename
        doc.metadata["source_file"] = filename

        # Human-readable page number
        doc.metadata["page_number"] = (
            doc.metadata.get("page", 0) + 1
        )

    return documents


# ============================================================
# 1. CHECK RTI ACT
# ============================================================

print("\n" + "=" * 60)
print("LEGAL RTI KNOWLEDGE BASE CREATION")
print("=" * 60)


if not os.path.exists(RTI_ACT_PATH):

    raise FileNotFoundError(
        f"\nRTI Act PDF not found:\n"
        f"{RTI_ACT_PATH}"
    )


# ============================================================
# 2. CHECK LAND CASE FOLDER
# ============================================================

if not os.path.exists(LAND_CASES_FOLDER):

    raise FileNotFoundError(
        f"\nLand-dispute cases folder not found:\n"
        f"{LAND_CASES_FOLDER}"
    )


# ============================================================
# 3. FIND ALL LAND CASE PDFs
# ============================================================

pdf_files = sorted(

    [
        filename
        for filename in os.listdir(
            LAND_CASES_FOLDER
        )
        if filename.lower().endswith(".pdf")
    ]

)


print("\nDocuments found")
print("-" * 60)

print(
    f"RTI Act: 1 PDF"
)

print(
    f"Land-dispute cases: {len(pdf_files)} PDFs"
)

print(
    f"Total PDFs: {len(pdf_files) + 1}"
)


if len(pdf_files) == 0:

    raise ValueError(
        "No land-dispute PDF files found."
    )


# ============================================================
# 4. LOAD RTI ACT
# ============================================================

print("\n" + "=" * 60)
print("1/2 — LOADING RTI ACT")
print("=" * 60)


all_documents = []


rti_documents = load_pdf(

    RTI_ACT_PATH,

    "RTI_ACT"

)


all_documents.extend(
    rti_documents
)


print(
    f"RTI Act pages loaded: "
    f"{len(rti_documents)}"
)


# ============================================================
# 5. LOAD ALL LAND-DISPUTE CASES
# ============================================================

print("\n" + "=" * 60)
print("2/2 — LOADING LAND-DISPUTE RTI CASES")
print("=" * 60)


successful_cases = 0
failed_cases = []


for index, filename in enumerate(

    pdf_files,

    start=1

):

    pdf_path = os.path.join(

        LAND_CASES_FOLDER,

        filename

    )


    print(
        f"\n[{index}/{len(pdf_files)}]"
    )


    try:

        documents = load_pdf(

            pdf_path,

            "LAND_RTI_CASE"

        )


        all_documents.extend(
            documents
        )


        successful_cases += 1


        print(
            f"    Pages loaded: "
            f"{len(documents)}"
        )


    except Exception as e:

        failed_cases.append(
            filename
        )


        print(
            f"    ERROR: {e}"
        )


# ============================================================
# 6. LOADING SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DOCUMENT LOADING SUMMARY")
print("=" * 60)


print(
    f"RTI Act pages: "
    f"{len(rti_documents)}"
)


print(
    f"Land cases successfully loaded: "
    f"{successful_cases}"
)


print(
    f"Land cases failed: "
    f"{len(failed_cases)}"
)


print(
    f"Total pages loaded: "
    f"{len(all_documents)}"
)


if failed_cases:

    print("\nFailed files:")

    for filename in failed_cases:

        print(
            f"  - {filename}"
        )


if len(all_documents) == 0:

    raise ValueError(
        "No documents were successfully loaded."
    )


# ============================================================
# 7. SPLIT DOCUMENTS
# ============================================================

print("\n" + "=" * 60)
print("CREATING TEXT CHUNKS")
print("=" * 60)


text_splitter = RecursiveCharacterTextSplitter(

    chunk_size=CHUNK_SIZE,

    chunk_overlap=CHUNK_OVERLAP,

    length_function=len

)


chunks = text_splitter.split_documents(

    all_documents

)


print(
    f"Chunk size: {CHUNK_SIZE}"
)


print(
    f"Chunk overlap: {CHUNK_OVERLAP}"
)


print(
    f"Total chunks created: "
    f"{len(chunks)}"
)


# ============================================================
# 8. SHOW DOCUMENT TYPE STATISTICS
# ============================================================

rti_chunks = 0
case_chunks = 0


for chunk in chunks:

    source_type = chunk.metadata.get(
        "source_type"
    )


    if source_type == "RTI_ACT":

        rti_chunks += 1


    elif source_type == "LAND_RTI_CASE":

        case_chunks += 1


print("\nChunk distribution:")

print(
    f"RTI Act chunks: {rti_chunks}"
)


print(
    f"Land-case chunks: {case_chunks}"
)


# ============================================================
# 9. CREATE EMBEDDING MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING EMBEDDING MODEL")
print("=" * 60)


print(
    f"Model: {EMBEDDING_MODEL}"
)


embedding = OllamaEmbeddings(

    model=EMBEDDING_MODEL

)


# ============================================================
# 10. CREATE FAISS DATABASE
# ============================================================

print("\n" + "=" * 60)
print("CREATING FAISS VECTOR DATABASE")
print("=" * 60)


vector_store = FAISS.from_documents(

    documents=chunks,

    embedding=embedding

)


print(
    f"Vectors created: "
    f"{vector_store.index.ntotal}"
)


# ============================================================
# 11. REMOVE OLD VECTOR DATABASE
# ============================================================

if os.path.exists(
    VECTORSTORE_PATH
):

    print(
        "\nRemoving old FAISS database..."
    )


    shutil.rmtree(
        VECTORSTORE_PATH
    )


# ============================================================
# 12. CREATE VECTORSTORE DIRECTORY
# ============================================================

os.makedirs(

    VECTORSTORE_PATH,

    exist_ok=True

)


# ============================================================
# 13. SAVE FAISS DATABASE
# ============================================================

print("\n" + "=" * 60)
print("SAVING FAISS DATABASE")
print("=" * 60)


vector_store.save_local(

    VECTORSTORE_PATH

)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("KNOWLEDGE BASE CREATED SUCCESSFULLY")
print("=" * 60)


print(
    f"\nFAISS location:"
    f"\n{VECTORSTORE_PATH}"
)


print(
    f"\nRTI Act:"
    f"\n  Pages: {len(rti_documents)}"
    f"\n  Chunks: {rti_chunks}"
)


print(
    f"\nLand-dispute RTI cases:"
    f"\n  PDFs: {successful_cases}"
    f"\n  Chunks: {case_chunks}"
)


print(
    f"\nTotal:"
    f"\n  Pages: {len(all_documents)}"
    f"\n  Chunks: {len(chunks)}"
    f"\n  Vectors: {vector_store.index.ntotal}"
)


if failed_cases:

    print(
        "\nWARNING:"
        f" {len(failed_cases)} files failed to load."
    )


print(
    "\nYour FAISS knowledge base is ready."
)
