# Interview Notes: LangChain RAG Assistant

## 60-second explanation

I built a document-based conversational RAG application with Streamlit and a Python backend. Users upload TXT, Markdown, PDF or DOCX documents. LangChain document loaders extract text, `RecursiveCharacterTextSplitter` creates overlapping chunks, `HuggingFaceEmbeddings` converts the chunks into vectors, and LangChain's Chroma integration stores them persistently. At query time, the application can rewrite follow-up questions, retrieves the most similar chunks, filters results by a configured distance threshold, builds a prompt using `ChatPromptTemplate`, and calls a Groq-hosted model through `ChatGroq`. The answer includes source references, while a JSON file preserves conversation history across restarts.

## Component roles

- `PyPDFLoader`, `TextLoader`, `Docx2txtLoader`: document ingestion.
- `RecursiveCharacterTextSplitter`: chunking.
- `HuggingFaceEmbeddings`: embedding generation.
- `Chroma`: vector persistence and similarity search.
- `ChatPromptTemplate`: structured prompts.
- `ChatGroq`: LangChain chat-model integration for Groq.
- `prompt | llm`: LCEL composition; the prompt output is passed to the model.
- `ConversationMemory`: custom JSON persistence. It is not LangChain's built-in conversation memory.
- Streamlit: frontend.

## Likely interview questions

### 1. What is RAG?
Retrieval-Augmented Generation retrieves relevant external context and supplies it to a language model so the answer can be grounded in the supplied information.

### 2. Why LangChain?
It standardizes common LLM application components such as loaders, splitters, embedding interfaces, vector stores, prompts and model integrations. It reduces glue code while leaving room for custom orchestration.

### 3. Why Chroma?
It is straightforward to run locally, supports persistent vector storage, metadata and similarity search, and integrates with LangChain.

### 4. Why embeddings?
Embeddings map text into vectors so semantically related questions and passages can be retrieved even when they do not share exact keywords.

### 5. What is chunk overlap?
Overlap repeats some text between adjacent chunks, helping preserve context around chunk boundaries.

### 6. What is top-k?
Top-k is the maximum number of candidate chunks retrieved for a query. This project uses four by default.

### 7. Does a distance threshold guarantee relevance?
No. It is a heuristic. The threshold must be validated against the chosen embedding model, distance metric and real documents.

### 8. How does the project cite sources?
It preserves `source` and, for PDF pages, page metadata in LangChain `Document` objects, then formats those labels alongside retrieved excerpts.

### 9. How does conversation memory work?
The app writes question, answer, timestamp, route and sources to a JSON file. It restores that file at startup. Recent turns may be included to resolve follow-up references.

### 10. Does this eliminate hallucinations?
No. It reduces risk through grounded prompts, retrieval filtering and a fallback when no chunk passes the threshold, but LLM output still needs evaluation.

### 11. What is LCEL?
LangChain Expression Language lets components be composed into runnable pipelines. In `prompt | llm`, the formatted prompt is passed to the chat model.

### 12. What would you improve?
Add retrieval evaluation, hybrid search, reranking, OCR, robust document identity, tests, authentication, per-user data isolation, structured logs, and a production database for conversation memory.
