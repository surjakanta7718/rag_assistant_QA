# LangChain RAG Assistant (Groq + ChromaDB + Streamlit)

A document-question-answering application using LangChain components for loading, splitting, embeddings, vector storage and prompt/model orchestration.

## Architecture

- **Frontend:** Streamlit (`app.py`)
- **LLM:** Groq API through `ChatGroq`
- **Embeddings:** `HuggingFaceEmbeddings` with `sentence-transformers/all-MiniLM-L6-v2`
- **Vector database:** `langchain-chroma` / ChromaDB, persisted to `rag_storage/chroma_db`
- **Loaders:** LangChain `PyPDFLoader`, `TextLoader`, `Docx2txtLoader`
- **Chunking:** `RecursiveCharacterTextSplitter`
- **Prompting:** `ChatPromptTemplate` and LCEL (`prompt | llm`)
- **Conversation persistence:** JSON file in `rag_storage/conversation_history.json`

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set your actual Groq API key:

```env
GROQ_API_KEY=your_real_key
GROQ_MODEL=openai/gpt-oss-120b
```

Use a model currently enabled for your Groq account. Do not commit `.env` to GitHub.

## Run

```powershell
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). Upload TXT, MD, PDF or DOCX documents from the sidebar, click **Index uploaded files**, then ask questions.

## RAG pipeline

1. LangChain document loaders extract text and metadata.
2. `RecursiveCharacterTextSplitter` splits documents into chunks of 800 characters with 150-character overlap.
3. `HuggingFaceEmbeddings` embeds chunks.
4. `Chroma` persists chunks and embeddings.
5. The user's question is optionally rewritten if it is a follow-up.
6. Chroma similarity search retrieves up to four chunks.
7. Results above the configured distance threshold are discarded.
8. `ChatPromptTemplate` combines recent conversation, document excerpts and the question.
9. `ChatGroq` generates the answer.
10. Sources and the exchange are saved.

## Main files

- `app.py`: Streamlit UI and calls to the backend.
- `rag_assistant.py`: configuration, loading, splitting, vector store, memory, retrieval and generation.
- `.env.example`: environment-variable template.
- `requirements.txt`: Python dependencies.

## Notes / limitations

- The distance threshold is a heuristic and should be evaluated against your actual documents. Chroma score semantics depend on the configured distance space and library version.
- A scanned/image-only PDF may need OCR; `PyPDFLoader` extracts selectable text but does not perform OCR.
- JSON memory is suitable for a local demo, not concurrent multi-user production. Consider a database for production.
- The trace is an application execution trace, not the model's private reasoning.
- For production, add authentication, upload size/type validation, rate limits, structured logging, tests, evaluation, and stronger document isolation.
