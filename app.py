"""Streamlit frontend for the LangChain RAG Assistant.
Run using: streamlit run app.py
"""  # Module-level docstring describing the application and how to launch it.

import os  # Provides functions to access environment variables and operating-system features.
from pathlib import Path  # Provides an object-oriented way to work with file and directory paths.

import streamlit as st  # Imports Streamlit to build the interactive web interface.
from dotenv import load_dotenv  # Imports a function to load environment variables from a .env file.

from rag_assistant import Config, RAGAssistant, SUPPORTED
# Config: stores the RAG configuration.
# RAGAssistant: implements the backend RAG pipeline.
# SUPPORTED: contains the supported document file extensions.

load_dotenv(Path(__file__).resolve().parent / ".env")
# Finds the directory containing this app.py file, constructs the path to .env,
# and loads environment variables from that file.
# By default, existing environment variables are not overwritten.

st.set_page_config(
    page_title="LangChain RAG Assistant",  # Sets the browser-tab title.
    page_icon="📚",  # Sets the page icon.
    layout="wide"  # Uses a wide layout to provide more horizontal space.
)

STORAGE = "./rag_storage"  # Defines the directory used to store application data.
UPLOADS = Path(STORAGE) / "uploads"  # Defines the directory where uploaded files are saved.

UPLOADS.mkdir(parents=True, exist_ok=True)
# Creates the uploads directory if it does not exist.
# parents=True creates missing parent directories; exist_ok=True avoids an error
# if the directory already exists.

if not os.getenv("GROQ_API_KEY"):  # Checks whether the Groq API key is available.
    st.error(
        "GROQ_API_KEY not found. Copy .env.example to .env and add your Groq API key."
    )  # Displays an error message in the Streamlit interface.
    st.stop()  # Stops execution so the application does not start without the API key.


@st.cache_resource(show_spinner="Loading embeddings and vector database...")
# Caches the returned assistant object across Streamlit reruns.
# This helps avoid repeatedly loading the embedding model and vector database.
def get_bot() -> RAGAssistant:  # Defines a function that returns the RAG assistant.
    return RAGAssistant(config=Config(persist_dir=STORAGE))
    # Creates the configuration with the storage directory and initializes the backend.


try:  # Begins error handling for assistant initialization.
    bot = get_bot()  # Gets the cached assistant or creates it on the first call.
except Exception as exc:  # Catches an exception if initialization fails.
    st.error(f"Could not start the assistant: {exc}")  # Shows the actual error in the UI.
    st.stop()  # Stops execution because the backend is unavailable.


with st.sidebar:  # Places the following interface elements in Streamlit's sidebar.
    st.header("Documents")  # Displays the Documents section heading.

    uploaded_files = st.file_uploader(
        "Upload files",  # Label displayed above the upload control.
        type=[ext.lstrip(".") for ext in sorted(SUPPORTED)],
        # Removes the leading dot from each extension, sorts the extensions,
        # and provides them as the allowed upload types.
        accept_multiple_files=True,  # Allows the user to upload multiple files.
    )

    if uploaded_files and st.button("Index uploaded files", type="primary"):
        # Checks that files have been uploaded and the indexing button was clicked.

        for uploaded in uploaded_files:  # Processes each uploaded file individually.

            # Reduce path traversal risk: use only the uploaded filename.
            safe_name = Path(uploaded.name).name
            # Extracts only the filename, excluding any directory components.

            destination = UPLOADS / safe_name
            # Creates the destination path inside the uploads directory.

            destination.write_bytes(uploaded.getvalue())
            # Reads the uploaded file's bytes and writes them to disk.

            with st.spinner(f"Indexing {safe_name}..."):
                # Displays a loading indicator while the file is processed.

                try:  # Starts error handling for document indexing.
                    count = bot.add_file(destination)
                    # Calls the backend to load, split, embed, and index the document.
                    # The returned count represents the number of indexed chunks.

                    st.success(f"{safe_name}: indexed {count} chunks")
                    # Displays a success message with the number of chunks.
                except Exception as exc:  # Handles errors while processing this file.
                    st.error(f"Could not index {safe_name}: {exc}")
                    # Displays the indexing error without hiding its details.

    stats = bot.stats()  # Gets statistics about indexed documents and conversation history.

    st.caption(
        f"{stats['indexed_chunks']} chunks • {stats['stored_exchanges']} saved exchanges"
    )  # Displays the total indexed chunks and saved conversation exchanges.

    st.caption(f"LLM: {stats['model']}")  # Displays the configured language model.
    st.caption(f"Embeddings: {stats['embedding_model']}")
    # Displays the embedding model used for document vectors.

    for filename, chunk_count in stats["documents"].items():
        # Iterates through indexed documents and their chunk counts.

        left, right = st.columns([4, 1])
        # Creates two columns: a wider column for the filename and a narrow one for deletion.

        left.write(f"📄 {filename} ({chunk_count})")
        # Displays the document name and number of chunks.

        if right.button(
            "✕",
            key=f"delete-{filename}",
            help=f"Delete indexed chunks for {filename}"
        ):
            # Creates a uniquely keyed delete button for each document.

            bot.delete_source(filename)
            # Removes the document's indexed chunks from the vector database.

            try:  # Handles possible errors when deleting the uploaded file.
                (UPLOADS / filename).unlink(missing_ok=True)
                # Deletes the uploaded file from disk if it exists.
            except OSError:  # Catches filesystem-related deletion errors.
                pass  # Ignores the filesystem error; the index deletion has already run.

            st.rerun()  # Reruns the app to refresh the document list.

    st.divider()  # Draws a horizontal divider in the sidebar.
    st.header("Conversation")  # Displays the conversation-management heading.

    if st.button("Clear history"):  # Checks whether the user clicked Clear history.
        bot.memory.clear()  # Clears the conversation memory maintained by the backend.
        st.rerun()  # Refreshes the interface after clearing the history.

    if bot.memory.history:  # Checks whether any conversation exchanges exist.

        export_path = bot.memory.export_markdown(Path(STORAGE) / "export.md")
        # Exports conversation history to a Markdown file and returns its path.

        st.download_button(
            "Download conversation (.md)",  # Label displayed on the download button.
            data=export_path.read_bytes(),  # Reads the generated Markdown file as bytes.
            file_name="conversation.md",  # Name given to the downloaded file.
            mime="text/markdown",  # Specifies the downloaded file's MIME type.
        )

    if st.button(
        "Reset index + history",
        help="Deletes all indexed vectors and conversation history"
    ):
        # Provides an option to reset the document index and conversation history.

        bot.reset_all()  # Calls the backend method to reset indexed data and memory.
        st.rerun()  # Refreshes the interface after the reset.


st.title("📚 LangChain RAG Assistant")
# Displays the main heading in the central application area.

st.caption(
    "Ask questions about uploaded documents. Answers include source references when available."
)
# Briefly explains the application's purpose.


for exchange in bot.memory.history:
    # Iterates through saved conversation exchanges to display previous messages.

    with st.chat_message("user"):  # Creates a chat bubble for the user's message.
        st.write(exchange["question"])  # Displays the previously asked question.

    with st.chat_message("assistant"):  # Creates a chat bubble for the assistant's answer.
        st.write(exchange["answer"])  # Displays the previously generated answer.

        if exchange.get("sources"):  # Checks whether the exchange contains source names.
            st.caption("Sources: " + ", ".join(exchange["sources"]))
            # Displays the source names as a comma-separated string.


question = st.chat_input("Ask about your documents or conversation...")
# Displays the chat input box and returns the submitted question, or None if empty.

if question:  # Executes this block only when the user submits a non-empty question.

    with st.chat_message("user"):  # Creates a chat bubble for the new question.
        st.write(question)  # Displays the user's submitted question.

    with st.chat_message("assistant"):  # Creates a chat bubble for the response.

        with st.spinner("Retrieving context and generating answer..."):
            # Displays a loading indicator while retrieval and answer generation run.

            try:  # Begins error handling for the RAG request.
                result = bot.ask(question)
                # Sends the question to the backend RAG pipeline.
                # The backend retrieves relevant chunks, builds the prompt,
                # invokes the language model, and returns the result.
            except Exception as exc:  # Catches errors during question processing.
                st.error(f"Request failed: {exc}")  # Displays the request error.
                st.stop()  # Stops execution after the failed request.

        st.write(result.answer)  # Displays the generated answer.

        if result.sources:  # Checks whether source references were returned.
            st.caption("Sources: " + ", ".join(result.sources))
            # Displays the source documents used for the answer.

        with st.expander(
            f"Execution trace • route: {result.route} • {result.latency_s}s"
        ):
            # Creates an expandable panel containing execution details.
            # route indicates the backend's selected processing route.
            # latency_s indicates the reported execution time in seconds.

            for item in result.trace:  # Iterates through the backend trace messages.
                st.text(item)  # Displays each trace item as plain text.