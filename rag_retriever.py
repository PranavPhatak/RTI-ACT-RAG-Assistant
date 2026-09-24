import os
import threading
from collections import defaultdict

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import WikipediaRetriever


# ============================================================
# CONFIGURATION
# ============================================================

VECTORSTORE_PATH = "vectorstore/land_rti_faiss"

# Embeddings stay local (Groq does not offer embedding models).
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

# FAISS returns a squared-L2 DISTANCE (lower = more similar).
# A chunk is treated as "relevant" only if its distance is <= this value.
# If NO chunk passes, the Wikipedia fallback is used.
# Tune it with RETRIEVAL_DEBUG=1 (see _retrieve()).
MAX_DISTANCE = float(os.getenv("FAISS_MAX_DISTANCE", "1.1"))

# When a metadata filter is used, FAISS first fetches `fetch_k` nearest
# chunks from the WHOLE index and only then applies the filter. The
# LangChain default (20) can return nothing for RTI-Act-only searches,
# so we look deeper.
FETCH_K = int(os.getenv("FAISS_FETCH_K", "200"))

# Wikipedia fallback settings
ENABLE_WIKIPEDIA = os.getenv("ENABLE_WIKIPEDIA_FALLBACK", "1") == "1"
WIKIPEDIA_TOP_K = int(os.getenv("WIKIPEDIA_TOP_K", "3"))
WIKIPEDIA_MAX_CHARS = int(os.getenv("WIKIPEDIA_MAX_CHARS", "3000"))

# Set RETRIEVAL_DEBUG=1 to print best FAISS distances in the console
DEBUG_RETRIEVAL = os.getenv("RETRIEVAL_DEBUG", "0") == "1"


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
# WIKIPEDIA FALLBACK
# ============================================================

def search_wikipedia(query, k=WIKIPEDIA_TOP_K):
    """
    Search Wikipedia and return LangChain Documents tagged with
    source_type = "WIKIPEDIA".

    Requires:  pip install wikipedia
    Never raises: on any failure it prints a warning and returns [].
    """

    query = (query or "").strip()[:300]

    if not query:
        return []

    try:

        retriever = WikipediaRetriever(
            top_k_results=k,
            lang="en",
            doc_content_chars_max=WIKIPEDIA_MAX_CHARS
        )

        docs = retriever.invoke(query)

    except Exception as error:

        print(
            f"[Wikipedia fallback] Search failed: {error}"
        )

        return []

    results = []

    for doc in docs:

        if not (doc.page_content or "").strip():
            continue

        title = doc.metadata.get(
            "title",
            "Wikipedia"
        )

        url = doc.metadata.get("source", "")

        if not str(url).startswith("http"):

            url = (
                "https://en.wikipedia.org/wiki/"
                + title.replace(" ", "_")
            )

        doc.metadata["source_type"] = "WIKIPEDIA"
        doc.metadata["source_file"] = f"Wikipedia: {title}"
        doc.metadata["page_number"] = None
        doc.metadata["url"] = url

        results.append(doc)

    return results


# ============================================================
# CORE RETRIEVAL (FAISS first, Wikipedia if nothing relevant)
# ============================================================

def _retrieve(query, k=5, source_type=None, use_wikipedia=True):

    query = (query or "").strip()

    if not query:
        return []

    search_kwargs = {}

    if source_type:

        search_kwargs["filter"] = {
            "source_type": source_type
        }

        search_kwargs["fetch_k"] = FETCH_K

    scored_docs = vector_store.similarity_search_with_score(
        query,
        k=k,
        **search_kwargs
    )

    relevant_docs = [
        doc
        for doc, score in scored_docs
        if score <= MAX_DISTANCE
    ]

    if DEBUG_RETRIEVAL:

        best = (
            f"{min(score for _, score in scored_docs):.3f}"
            if scored_docs
            else "none"
        )

        print(
            f"[Retrieval] type={source_type or 'ALL'} "
            f"best_distance={best} "
            f"threshold={MAX_DISTANCE} "
            f"relevant={len(relevant_docs)}/{len(scored_docs)}"
        )

    if relevant_docs:
        return relevant_docs

    # Nothing relevant in FAISS -> Wikipedia fallback
    if use_wikipedia and ENABLE_WIKIPEDIA:

        wiki_docs = search_wikipedia(query)

        if DEBUG_RETRIEVAL:
            print(
                f"[Retrieval] Wikipedia fallback returned "
                f"{len(wiki_docs)} document(s)"
            )

        return wiki_docs

    return []


# ============================================================
# RETRIEVE FROM RTI ACT
# ============================================================

def retrieve_rti_act(query, k=5, use_wikipedia=True):

    return _retrieve(
        query,
        k=k,
        source_type="RTI_ACT",
        use_wikipedia=use_wikipedia
    )


# ============================================================
# RETRIEVE LAND-DISPUTE CASES
# ============================================================

def retrieve_land_cases(query, k=5, use_wikipedia=True):

    return _retrieve(
        query,
        k=k,
        source_type="LAND_RTI_CASE",
        use_wikipedia=use_wikipedia
    )


# ============================================================
# RETRIEVE FROM EVERYTHING
# ============================================================

def retrieve_all(query, k=5, use_wikipedia=True):

    return _retrieve(
        query,
        k=k,
        source_type=None,
        use_wikipedia=use_wikipedia
    )


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

        if source_type == "WIKIPEDIA":

            formatted.append(
                f"""
Source Type: WIKIPEDIA
Source: {source_file}
URL: {doc.metadata.get("url", "")}
Note: General background from Wikipedia. This is NOT an
authoritative legal text and NOT a court case.

Content:
{doc.page_content}
"""
            )

            continue

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


# ============================================================
# CONVERSATION MEMORY
# ============================================================

class ConversationMemory:
    """
    Lightweight, thread-safe, per-session conversation memory.

    * The Orchestrator owns one instance (whole conversation).
    * EVERY agent owns its own instance (what that agent was asked
      and what it produced earlier in this conversation).

    Memory is keyed by `session_id`, so several Streamlit users
    sharing one cached Orchestrator never see each other's history.
    """

    def __init__(self, max_turns=20, max_chars=1500):

        self.max_turns = max_turns

        self.max_chars = max_chars

        self._turns = defaultdict(list)

        self._lock = threading.Lock()

    # --------------------------------------------------------

    def _clip(self, text):

        text = str(text or "").strip()

        if len(text) <= self.max_chars:
            return text

        return (
            text[: self.max_chars].rstrip()
            + " ...[truncated]"
        )

    # --------------------------------------------------------

    def add(self, session_id, user, assistant, **meta):

        with self._lock:

            turns = self._turns[session_id]

            turns.append(
                {
                    "user": self._clip(user),
                    "assistant": self._clip(assistant),
                    "meta": meta
                }
            )

            if len(turns) > self.max_turns:

                del turns[: len(turns) - self.max_turns]

    # --------------------------------------------------------

    def get(self, session_id, last_n=None):

        with self._lock:

            turns = list(
                self._turns.get(session_id, [])
            )

        if last_n:
            return turns[-last_n:]

        return turns

    # --------------------------------------------------------

    def as_text(
        self,
        session_id,
        last_n=None,
        user_label="USER",
        assistant_label="ASSISTANT",
        empty="No previous interactions."
    ):

        turns = self.get(session_id, last_n)

        if not turns:
            return empty

        lines = []

        for turn in turns:

            lines.append(
                f"{user_label}: {turn['user']}"
            )

            lines.append(
                f"{assistant_label}: {turn['assistant']}"
            )

            lines.append("")

        return "\n".join(lines).strip()

    # --------------------------------------------------------

    def clear(self, session_id=None):

        with self._lock:

            if session_id is None:

                self._turns.clear()

            else:

                self._turns.pop(session_id, None)
