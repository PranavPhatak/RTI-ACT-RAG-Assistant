# 📜 RTI Act RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about the **Right to Information Act, 2005** — built with Qwen3 embeddings, FAISS, and Llama 3, served through a Streamlit UI.

---

## 🧠 Overview

Instead of relying purely on an LLM's memorized knowledge, this assistant retrieves the most relevant sections of the actual RTI Act PDF and grounds its answers in that retrieved context — reducing hallucination and keeping answers legally accurate.

---

## 🔄 How It Works

```
                 RTI Act PDF
                      │
                      ▼
                PDF Loader
                      │
                      ▼
                Text Chunks
                      │
                      ▼
             Qwen3 Embeddings
                      │
                      ▼
                  FAISS
              Vector Database
                      │
                      ▼
                 Retriever
                      │
            Relevant Documents
                      │
                      ▼
                  Prompt
                      │
                      ▼
                 Llama 3
                      │
                      ▼
                Final Answer
```

1. **PDF Loader** — parses `RTI-Act_English.pdf` into raw text.
2. **Text Chunks** — the document is split into overlapping chunks for retrieval.
3. **Qwen3 Embeddings** — each chunk is embedded into a vector representation.
4. **FAISS Vector Database** — stores the embeddings for fast similarity search.
5. **Retriever** — given a user query, fetches the most relevant chunks.
6. **Prompt** — retrieved documents + user question are assembled into a grounded prompt.
7. **Llama 3** — generates the final answer using the retrieved context.

---

## 📂 Project Structure

```
RTI-ACT-RAG-ASSISTANT/
├── app.py                  # Streamlit UI
├── main.py                 # Entry point / orchestration
├── rag.py                  # RAG pipeline (loader, embeddings, FAISS, retriever, LLM chain)
├── RTI-Act_English.pdf     # Source document (RTI Act, 2005)
├── pyproject.toml          # Project metadata & dependencies
├── uv.lock                 # Locked dependency versions (uv package manager)
├── requirements.txt        # Pip-installable dependency list
├── .python-version         # Pinned Python version
├── README.md
└── LICENSE
```

---

## ⚙️ Tech Stack

| Component        | Technology            |
|-------------------|------------------------|
| UI                | Streamlit              |
| Embeddings        | Qwen3                  |
| Vector Store      | FAISS                  |
| LLM               | Llama 3                |
| Package Manager   | uv                     |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd RTI-ACT-RAG-ASSISTANT
```

### 2. Set up the environment
Using `uv` (recommended):
```bash
uv sync
```

Or using `pip`:
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## 💬 Usage

1. Launch the app.
2. Type a question about the RTI Act, 2005 (or pick one of the example questions in the sidebar).
3. Click **Ask** to retrieve a grounded answer sourced from the Act itself.
4. Previous questions and answers are kept in the conversation history for the session.

**Example questions:**
- Who can file an RTI application?
- What is the time limit to respond to an RTI request?
- What information is exempt from disclosure?
- What is the fee for filing an RTI application?

---

## 📄 License

See [LICENSE](./LICENSE) for details.

---

## 🙏 Acknowledgements

- Right to Information Act, 2005 — Government of India
- Qwen3 embedding models
- Meta's Llama 3
- FAISS by Meta AI Research
